"""
wet_ink.py v2 — training-free "wet-ink" decoding for Nemotron-Labs-Diffusion.

Concept (user's idea)
---------------------
The diffusion frontier opens tokens left-to-right. Already-opened ("committed")
tokens inside a bounded revision window behind the frontier stay correctable
while the frontier writes the next block; ink far from the frontier dries
(leaves the window). No retraining.

What changed in v2 and why (root cause of the v1 problem)
---------------------------------------------------------
v1 applied the LLaDA confidence test to OPEN window tokens at EVERY denoising
step while keeping the stock threshold-based commit loop. That composition has
no progress guarantee: re-masking can add masks faster than the forced top-1
commit removes them, so the loop only ended at a hard cap + safety tail
(measured on the 3B: 9-19x stock NFE, 57% of blocks hitting the cap, ~589
remask events per block for ~0.4 real token changes). The formulas were
published and verbatim; the missing piece was a SYSTEM INVARIANT, not a
formula. Every published method that revises open tokens ships one:

  * RemeDi  (ICLR 2026, arXiv:2509.23653) — fixed step count; each step the
    open set is rebuilt as the cumulative top-k by confidence, everything
    else (including previously open tokens) returns to [MASK]. Terminates by
    construction: the cumulative budget reaches the span length at the last
    step, so every position ends up kept.
  * WINO    (Feng-Hong/WINO-DLLM) — threshold loop with the coupling
    remask_budget <= accept_budget - 1: net progress >= 1 per step.
  * Saber   (ACL 2026, arXiv:2510.18165) — exchange rule
    current_remask_k = min(unmask_k, num_unmasked): swap the k worst open
    tokens for k fresh predictions.
  * CoDR    (EMNLP 2026, arXiv:2610.08833) — stock denoise loop untouched;
    AFTER each block, probe committed positions (k-partition probing), remask
    those whose confidence DROPPED (drift = old_conf - new_conf >
    drift_threshold) and refill with the same sampler.

v2 implements the two mechanisms that compose most cleanly with the stock
Nemotron generator (both training-free, both with published termination
guarantees):

  remask_policy='reme_di' (default) — RemeDi's inference loop over the span
      [W completed blocks + current block] with stock scoring formulas.
  remask_policy='codr' — stock block loop + CoDR post-block drift pass over
      the revision window.

'per_step' (v1 behaviour) is kept for reproducibility of the v1 measurements
and is documented as superseded. 'none' is bit-identical to stock.

Provenance of every mechanism (NO new formulas, NO modified formulas)
---------------------------------------------------------------------
Stock references are to modeling_nemotron_labs_diffusion.py of
nvidia/Nemotron-Labs-Diffusion-3B (HF, sha-matched to the public repo). The
model's own helpers are imported from the loaded model module at runtime.

1. Commit rule — stock ``_get_transfer_index`` (L835-861), called verbatim:
   Gumbel-argmax x0 (L842-843), confidence x0_p = softmax(logits)[x0]
   (L845-846; identical formula to LLaDA generate.py and to
   ``sampler.py::_unmask_low_confidence`` of the NVlabs repo), top-k budget +
   optional threshold filter with the top-1 always committed (L852-860).
   Budget split: stock ``_get_num_transfer_tokens`` (L824-832) = LLaDA
   ``get_num_transfer_tokens``.

2. RemeDi loop (remask_policy='reme_di') — inference algorithm of
   maple-research-lab/RemeDi (official implementation of arXiv:2509.23653),
   inference.py L64-90, verbatim semantics:
       num_transfer_tokens = even split of span positions over steps; cumsum
       keep = topk(score, k=num_transfer_tokens[i])           [their L84]
       x = where(keep, x0, [MASK])                             [their L90]
   Composition details (documented, no formula changes):
     * the span is [W revision blocks + current block] instead of one block —
       our window extension; the KV cache is cropped with the stock
       ``_crop_dynamic_cache`` (L803-811; the helper ``linear_spec_generate``
       uses for the same purpose at L775) and rebuilt by the stock post-block
       causal forward;
     * the score is the stock softmax confidence x0_p (L845-846) with the
       stock x0-substitution (L848: open positions keep their current token)
       instead of RemeDi's raw-logit gather — the model's own published
       formula in the same role; optionally the Dream/NVlabs negative-entropy
       score (``sampler.py`` of the NVlabs repo: sum(p*log(p+eps)) over the
       vocabulary, eps=1e-10) for criterion='entropy';
     * the stock AR seed (block position 0 = next_token) is pinned in the
       kept set — stock never re-predicts that position (it is written before
       the denoise loop and is never a mask), so pinning preserves stock
       semantics;
     * steps = block_length by default (the stock "one denoising step per
       generated token" rule, L412); the budget split is taken over the span
       length, following RemeDi's split of block positions over steps.
   State-space validity: training masks i.i.d. Bernoulli over all positions
   (L220-232), so any mask subset is in-distribution; the fully bidirectional
   attention of the shipped 3B paradigm has no span-boundary constraints.

3. CoDR pass (remask_policy='codr') — YueWu0301/CoDR (official implementation
   of arXiv:2610.08833), codr/generation.py, verbatim semantics:
       probe: mask one partition of committed positions, forward, read
              new_confidence = softmax(logits)[current token] at the masked
              positions                                  [their L141-153]
       drift: old_confidence - new_confidence > drift_threshold
              -> remask                                   [their L155]
       refill: the same sampler loop over the remasked positions [their L159-170]
       old_confidence is stored at commit time            [their L135]
   Composition details: probing is restricted to the revision window (the
   wet-ink span), probe forwards run against the cropped stock KV cache,
   refill uses the stock ``_get_transfer_index`` threshold path.

4. 'per_step' (v1, superseded) — a second verbatim call of stock
   ``_get_transfer_index`` with mask_index = open window positions; its
   threshold filter implements the model's own confidence test
   ``draft_conf >= threshold`` (L727-736, linear_spec_generate). Kept for
   reproducibility; has no progress guarantee (see header).

Everything else below is control flow around these published primitives.
"""

import importlib
from typing import Optional

import torch
import torch.nn.functional as F


POLICIES = ("reme_di", "codr", "per_step", "none")
CRITERIA = ("confidence", "entropy")


def get_stock_helpers(model):
    """Return the model's OWN helper functions from the remote-code module
    transformers loaded for this checkpoint, so every formula is executed from
    the original source file."""
    module = importlib.import_module(type(model).__module__)
    return (
        module._get_transfer_index,
        module._get_num_transfer_tokens,
        module._crop_dynamic_cache,
        module._add_gumbel_noise,
    )


def _span_scores(logits, mask_idx, span_ids, temperature, criterion, _add_gumbel_noise):
    """Score every span position with published formulas.

    Returns (x0, score):
      x0    — Gumbel-argmax prediction at masked positions, current token at
              open positions  [stock _get_transfer_index L842-848, verbatim]
      score — confidence x0_p = softmax(logits)[x0]   [stock L845-846; LLaDA
              generate.py L180-183; RemeDi upm semantics for open positions]
              or negative entropy  [NVlabs sampler.py / Dream 7B:
              sum(p * log(p + eps)) over the vocabulary, eps = 1e-10]
    """
    logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_noisy, dim=-1)
    x0 = torch.where(mask_idx, x0, span_ids)  # stock L848
    p = F.softmax(logits.float(), dim=-1)
    if criterion == "entropy":
        logp = torch.log(p + 1e-10)
        score = torch.sum(p * logp, dim=-1)  # negative entropy per position
    else:
        score = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )
    return x0, score


@torch.no_grad()
def generate_wet_ink(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    block_length: int,
    revision_blocks: int = 1,
    remask_policy: str = "reme_di",
    criterion: str = "confidence",
    threshold: Optional[float] = 0.9,
    drift_threshold: float = 0.0,
    num_partitions: int = 2,
    denoise_steps: Optional[int] = None,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    causal_context: bool = True,
    commit: str = "threshold",   # "threshold" (stock) | "fixed1" | "budget"
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    return_stats: bool = False,
):
    """Stock block-diffusion ``generate()`` + bounded revision window.

    remask_policy:
      'reme_di' — RemeDi loop (ICLR 2026) over the span: every step the open
                  set is rebuilt as the cumulative top-k by score; all other
                  span positions (incl. previously open) return to [MASK].
                  Exactly ``denoise_steps`` forwards per block, no tail, no
                  cap — termination by construction.
      'codr'    — stock loop + post-block CoDR pass (EMNLP 2026): probe the
                  window partitions, remask confidence-drifted tokens,
                  refill with the stock sampler.
      'per_step'— v1 behaviour (superseded; no progress guarantee).
      'none'    — bit-identical to stock generate().

    criterion: 'confidence' (stock x0_p) or 'entropy' (Dream/NVlabs negative
      entropy) — the ranking score for 'reme_di'.

    threshold: commit threshold for the stock/codr/per_step loops (stock
      default 0.9); also the CoDR refill threshold. Not used by 'reme_di'
      (rank-based selection, threshold-free).

    Returns (output_ids, nfe) like stock; with return_stats=True also a stats
    dict. NFE counted exactly like stock generate() (denoise steps +
    post-block forwards; prefill not counted, matching stock L414/L476/L511).
    """
    assert remask_policy in POLICIES, remask_policy
    assert criterion in CRITERIA, criterion
    if revision_blocks == 0 and remask_policy != "none":
        # v1 contract preserved: revision_blocks=0 is ALWAYS the stock path
        # (bit-identical to stock generate()), regardless of the policy.
        remask_policy = "none"
    (_get_transfer_index, _get_num_transfer_tokens, _crop_dynamic_cache,
     _add_gumbel_noise) = get_stock_helpers(model)

    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    mask_id = model.mask_token_id

    x_accum = prompt_ids.clone()
    B = prompt_ids.shape[0]
    prompt_len = prompt_ids.size(1)
    device = prompt_ids.device

    assert max_new_tokens % block_length == 0
    num_blocks = max_new_tokens // block_length
    if denoise_steps is None:
        denoise_steps = block_length  # stock rule: one step per generated token

    nfe = 0
    stats = {
        "policy": remask_policy,
        "criterion": criterion,
        "denoise_forwards": 0,
        "post_block_forwards": 0,
        "commit_events": 0,
        "remask_events": 0,
        "blocks_with_remask": 0,
        "revision_token_changes": 0,
        "safety_tail_steps": 0,
        "codr_probe_forwards": 0,
        "codr_drift_remasks": 0,
        "codr_refill_forwards": 0,
        "hard_cap_hits": 0,
        "per_block": [],
    }
    # CoDR: commit-time confidence of every position [CoDR generation.py L115]
    old_confidence = torch.zeros_like(x_accum, dtype=torch.float32)

    def _set_diffusion_lm(val: bool):
        for layer in model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = val

    def _stock_eos_fired(block_slice):
        """Stock EOS early-stop test [generate L491-498, verbatim]: EOS
        appears in the current block with no [MASK] before it."""
        block_tokens = x_accum[:, block_slice]
        eos_mask = block_tokens == eos_token_id
        after_eos = eos_mask.cumsum(dim=1).bool()
        mask_before = (block_tokens == mask_id) & ~after_eos
        return bool((eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any())

    # Initial causal prefill -> KV cache + next-block seed. [stock L421-435]
    if causal_context:
        _set_diffusion_lm(False)
    output = model(prompt_ids, use_cache=True, use_causal_mask=causal_context)
    past_key_values = output.past_key_values
    if causal_context:
        _set_diffusion_lm(True)

    next_token = None
    if causal_context:
        last_logit = output.logits[:, -1, :]
        if temperature > 0:
            next_token = torch.multinomial(
                torch.softmax(last_logit / temperature, dim=-1), num_samples=1
            )
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_slice = slice(block_start, block_start + block_length)
        rev_start = prompt_len + max(0, num_block - revision_blocks) * block_length
        span_slice = slice(rev_start, block_start + block_length)
        span_len = span_slice.stop - span_slice.start
        window_len = block_start - rev_start  # 0 => no revision this block

        mask_block = torch.full(
            (B, block_length), mask_id, dtype=prompt_ids.dtype, device=device,
        )
        if causal_context:
            mask_block[:, 0] = next_token[:, 0]
        x_accum = torch.cat([x_accum, mask_block], dim=1)
        if old_confidence.shape[1] < x_accum.shape[1]:
            old_confidence = torch.cat(
                [old_confidence,
                 torch.zeros((B, x_accum.shape[1] - old_confidence.shape[1]),
                             dtype=torch.float32, device=device)],
                dim=1,
            )

        # Thinking-budget enforcement. [stock L448-465, verbatim]
        if end_think_token_id is not None and max_thinking_tokens is not None:
            tokens_before = num_block * block_length
            tokens_after = tokens_before + block_length
            if tokens_after > max_thinking_tokens:
                gen_so_far = x_accum[:, prompt_len:block_start]
                has_end_think = (
                    (gen_so_far == end_think_token_id).any(dim=1)
                    if gen_so_far.size(1) > 0
                    else torch.zeros(B, dtype=torch.bool, device=device)
                )
                if not has_end_think.all():
                    offset = max(0, max_thinking_tokens - tokens_before)
                    inject_pos = block_start + offset
                    for b in range(B):
                        if not has_end_think[b]:
                            x_accum[b, inject_pos] = end_think_token_id

        window_entry = (
            x_accum[:, rev_start:block_start].clone() if window_len > 0 else None
        )

        # Crop the KV cache to the span start. [stock _crop_dynamic_cache,
        # same role as in linear_spec_generate L775]
        if window_len > 0:
            _crop_dynamic_cache(past_key_values, rev_start)

        remasked_in_block = 0
        stopped_by_eos = False

        # ---- policy: RemeDi loop [RemeDi inference.py L64-90] ----
        if remask_policy == "reme_di":
            # RemeDi budget: even split of SPAN POSITIONS over steps + cumsum
            # (their L64-68). We obtain the split from the stock helper over
            # an all-ones "mask", so the split formula itself is the model's
            # own published one.
            ones_span = torch.ones((B, span_len), dtype=torch.bool, device=device)
            k_per_step = _get_num_transfer_tokens(ones_span, denoise_steps)
            k_cum = k_per_step.cumsum(dim=1)  # RemeDi cumulative usage
            seed_pos = window_len  # block position 0 inside the span

            for i in range(denoise_steps):
                span_ids = x_accum[:, span_slice]
                mask_span_idx = span_ids == mask_id
                nfe += 1
                stats["denoise_forwards"] += 1
                logits_span = model(
                    span_ids, past_key_values=past_key_values, use_cache=False,
                ).logits
                x0, score = _span_scores(
                    logits_span, mask_span_idx, span_ids,
                    temperature, criterion, _add_gumbel_noise,
                )
                keep = torch.zeros_like(mask_span_idx)
                for b in range(B):
                    _, idx = torch.topk(score[b], k=int(k_cum[b, i]))
                    keep[b, idx] = True
                keep[:, seed_pos] = True  # stock AR seed is never re-predicted
                n_remask = int((~keep & ~mask_span_idx).sum().item())
                remasked_in_block += n_remask
                stats["remask_events"] += n_remask
                stats["commit_events"] += int((keep & mask_span_idx).sum().item())
                # RemeDi L90: x = where(keep, x0, [MASK])
                x_accum[:, span_slice] = torch.where(
                    keep, x0, torch.full_like(x0, mask_id)
                )
                if eos_token_id is not None and _stock_eos_fired(block_slice):
                    stopped_by_eos = True
                    break

        # ---- policy: stock loop (+ v1 per_step remasking, + codr hooks) ----
        else:
            mask_span_idx0 = x_accum[:, span_slice] == mask_id
            steps_cap = span_len  # one denoising step per span token (stock rule)
            num_transfer_tokens = _get_num_transfer_tokens(mask_span_idx0, steps_cap)
            commit_threshold = threshold if commit == "threshold" else None
            # 'fixed1' = sampler.py 'fixed' with k=1; 'budget' = LLaDA forced-k
            # (threshold=None) with the stock schedule.

            def _budget_for_step(mask_idx_now, i):
                if commit == "fixed1":
                    return torch.ones(B, dtype=torch.long, device=device)
                if commit_threshold is not None:
                    return num_transfer_tokens[:, min(i, steps_cap - 1)]
                if commit == "budget":
                    return num_transfer_tokens[:, min(i, steps_cap - 1)]
                remaining = max(1, steps_cap - i)
                return _get_num_transfer_tokens(mask_idx_now, remaining)[:, 0]

            hard_cap = 4 * steps_cap  # control-flow bound (per_step legacy only)

            for i in range(hard_cap):
                span_ids = x_accum[:, span_slice]
                mask_span_idx = span_ids == mask_id
                if mask_span_idx.sum() == 0:
                    break
                nfe += 1
                stats["denoise_forwards"] += 1
                logits_span = model(
                    span_ids, past_key_values=past_key_values, use_cache=False,
                ).logits
                budget_i = _budget_for_step(mask_span_idx, i)
                x0, transfer_idx = _get_transfer_index(
                    logits_span, temperature, mask_span_idx, span_ids,
                    num_transfer_tokens=budget_i, threshold=commit_threshold,
                )
                # CoDR: remember commit-time confidence [generation.py L135]
                if remask_policy == "codr":
                    p = F.softmax(logits_span.float(), dim=-1)
                    conf = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                    old_confidence[:, span_slice] = torch.where(
                        transfer_idx, conf, old_confidence[:, span_slice]
                    )
                cur = span_ids.clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, span_slice] = cur
                stats["commit_events"] += int(transfer_idx.sum().item())

                # ---- v1 per-step remasking (superseded; kept for repro) ----
                if (remask_policy == "per_step" and window_len > 0
                        and threshold is not None):
                    new_block_mask_idx = x_accum[:, block_slice] == mask_id
                    if bool(new_block_mask_idx.any()):
                        win_ids = x_accum[:, rev_start:block_start]
                        win_open = win_ids != mask_id
                        if bool(win_open.any()):
                            _, keep_idx = _get_transfer_index(
                                logits_span[:, :window_len], temperature,
                                win_open, win_ids,
                                num_transfer_tokens=None, threshold=threshold,
                            )
                            re_mask = win_open & ~keep_idx
                            if bool(re_mask.any()):
                                cur = x_accum[:, span_slice].clone()
                                cur[:, :window_len][re_mask] = mask_id
                                x_accum[:, span_slice] = cur
                                n_rm = int(re_mask.sum().item())
                                remasked_in_block += n_rm
                                stats["remask_events"] += n_rm

                if eos_token_id is not None and _stock_eos_fired(block_slice):
                    stopped_by_eos = True
                    break

            if remask_policy == "per_step":
                if i == hard_cap - 1 and not stopped_by_eos:
                    stats["hard_cap_hits"] += 1
                # v1 safety tail (control flow only): clear remaining masks
                # with commit-only steps before the cache rebuild.
                if not stopped_by_eos:
                    for _ in range(span_len):
                        span_ids = x_accum[:, span_slice]
                        mask_span_idx = span_ids == mask_id
                        if mask_span_idx.sum() == 0:
                            break
                        nfe += 1
                        stats["denoise_forwards"] += 1
                        stats["safety_tail_steps"] += 1
                        logits_span = model(
                            span_ids, past_key_values=past_key_values,
                            use_cache=False,
                        ).logits
                        budget_i = (
                            torch.ones(B, dtype=torch.long, device=device)
                            if commit == "fixed1"
                            else _get_num_transfer_tokens(mask_span_idx, span_len)[:, 0]
                        )
                        x0, transfer_idx = _get_transfer_index(
                            logits_span, temperature, mask_span_idx, span_ids,
                            num_transfer_tokens=budget_i, threshold=commit_threshold,
                        )
                        cur = span_ids.clone()
                        cur[transfer_idx] = x0[transfer_idx]
                        x_accum[:, span_slice] = cur
                        stats["commit_events"] += int(transfer_idx.sum().item())
                        if eos_token_id is not None and _stock_eos_fired(block_slice):
                            break

        # ---- CoDR pass: post-block probe + drift remask + refill ----
        # [CoDR generation.py L140-170; restricted to the revision window]
        if remask_policy == "codr" and window_len > 0 and not stopped_by_eos:
            span_ids = x_accum[:, span_slice]
            if not bool((span_ids == mask_id).any()):
                win_pos = torch.arange(window_len, device=device)
                new_conf = torch.zeros((B, window_len), dtype=torch.float32,
                                       device=device)
                for q in range(num_partitions):
                    part_idx = win_pos[(win_pos % num_partitions) == q]
                    if part_idx.numel() == 0:
                        continue
                    probe = span_ids.clone()
                    probe[:, part_idx] = mask_id  # CoDR k-partition probing
                    logits_probe = model(
                        probe, past_key_values=past_key_values, use_cache=False,
                    ).logits
                    nfe += 1
                    stats["codr_probe_forwards"] += 1
                    p = F.softmax(logits_probe.float(), dim=-1)
                    cur_tok = span_ids[:, part_idx]
                    conf_part = torch.squeeze(
                        torch.gather(
                            p[:, part_idx], dim=-1,
                            index=torch.unsqueeze(cur_tok, -1),
                        ), -1,
                    )
                    new_conf[:, part_idx] = conf_part
                old_win = old_confidence[:, rev_start:block_start]
                drift = (old_win - new_conf) > drift_threshold  # CoDR formula
                if bool(drift.any()):
                    cur = x_accum[:, span_slice].clone()
                    cur[:, :window_len][drift] = mask_id
                    x_accum[:, span_slice] = cur
                    n_dr = int(drift.sum().item())
                    remasked_in_block += n_dr
                    stats["remask_events"] += n_dr
                    stats["codr_drift_remasks"] += n_dr
                    # refill [CoDR L159-170] — stock commits, bounded
                    for _ in range(span_len):
                        span_ids = x_accum[:, span_slice]
                        mask_span_idx = span_ids == mask_id
                        if mask_span_idx.sum() == 0:
                            break
                        nfe += 1
                        stats["denoise_forwards"] += 1
                        stats["codr_refill_forwards"] += 1
                        logits_span = model(
                            span_ids, past_key_values=past_key_values,
                            use_cache=False,
                        ).logits
                        budget_i = _get_num_transfer_tokens(
                            mask_span_idx, span_len
                        )[:, 0]
                        x0, transfer_idx = _get_transfer_index(
                            logits_span, temperature, mask_span_idx, span_ids,
                            num_transfer_tokens=budget_i, threshold=threshold,
                        )
                        p = F.softmax(logits_span.float(), dim=-1)
                        conf = torch.squeeze(
                            torch.gather(p, dim=-1,
                                         index=torch.unsqueeze(x0, -1)), -1
                        )
                        old_confidence[:, span_slice] = torch.where(
                            transfer_idx, conf, old_confidence[:, span_slice]
                        )
                        cur = span_ids.clone()
                        cur[transfer_idx] = x0[transfer_idx]
                        x_accum[:, span_slice] = cur
                        stats["commit_events"] += int(transfer_idx.sum().item())
                        if eos_token_id is not None and _stock_eos_fired(block_slice):
                            break

        # Post-block: causal forward over the WHOLE span to rebuild the KV
        # cache and (when causal_context) sample the next seed.
        # [stock L500-519; span-extended so revised tokens re-enter the cache]
        if causal_context:
            _set_diffusion_lm(False)
        output = model(
            x_accum[:, span_slice],
            past_key_values=past_key_values,
            use_cache=True,
            use_causal_mask=causal_context,
        )
        past_key_values = output.past_key_values
        nfe += 1
        stats["post_block_forwards"] += 1

        if causal_context:
            _set_diffusion_lm(True)
            last_logit = output.logits[:, -1, :]
            if temperature > 0:
                next_token = torch.multinomial(
                    torch.softmax(last_logit / temperature, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        if window_len > 0 and window_entry is not None:
            now = x_accum[:, rev_start:block_start]
            stats["revision_token_changes"] += int((now != window_entry).sum().item())
        if remasked_in_block > 0:
            stats["blocks_with_remask"] += 1
        stats["per_block"].append(
            dict(
                block=num_block,
                window_len=window_len,
                remasked=remasked_in_block,
                revised_tokens=(
                    int((x_accum[:, rev_start:block_start] != window_entry).sum().item())
                    if window_entry is not None
                    else 0
                ),
            )
        )

        # EOS final check. [stock L521-527, verbatim]
        if eos_token_id is not None:
            gen_so_far = x_accum[:, prompt_len:]
            is_eos = gen_so_far == eos_token_id
            if is_eos.any(dim=1).all():
                first_eos = is_eos.to(torch.int64).argmax(dim=1)
                max_eos = first_eos.max().item()
                out = x_accum[:, : prompt_len + max_eos + 1]
                if return_stats:
                    stats["_final_cache"] = past_key_values
                    return out, nfe, stats
                return out, nfe

    if return_stats:
        stats["_final_cache"] = past_key_values
        return x_accum, nfe, stats
    return x_accum, nfe
# ====================== KAGGLE HARNESS v2 (appended after wet_ink.py) ======================
# Experiment: stock block-diffusion decoding vs PUBLISHED bounded-remasking
# mechanisms (RemeDi ICLR'26 loop, CoDR EMNLP'26 drift pass) on the FULL
# nvidia/Nemotron-Labs-Diffusion-3B (no retraining).
#
#   1. sanity chat (stock generate, chat defaults block_length=32, threshold=0.9)
#   2. equivalence: generate_wet_ink(revision_blocks=0) must reproduce stock
#      generate() bit-for-bit (ids AND nfe) at temperature 0 — for every policy
#   3. qualitative: side-by-side outputs + revision statistics on 3 prompts
#   4. GSM8K A/B (protocol mirrors the official evaluate.py):
#        stock                      — baseline (v1-comparable)
#        remedi_w1_s32_conf         — RemeDi loop, stock step density
#        remedi_w1_s16_conf         — RemeDi loop, halved steps
#        remedi_w1_s16_entropy      — RemeDi loop + Dream entropy criterion
#        codr_w1_t0.9               — CoDR drift pass after each block
#        remedi_w2_s16_conf         — window scaling W=2 (if time allows)
#
# All results -> /kaggle/working/results.json (+ this log).

import json
import os
import re
import subprocess
import sys
import time

def _pip(*pkgs):
    print("pip install:", *pkgs, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

_pip("transformers==5.0.0", "datasets")

import torch  # noqa: E402
import transformers  # noqa: E402
from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

MODEL_ID = "nvidia/Nemotron-Labs-Diffusion-3B"
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")
TIME_LIMIT_S = 5.5 * 3600  # keep well inside the kernel limit
T0 = time.time()

RESULTS = {"meta": {}, "sanity": {}, "equivalence": {}, "qualitative": [], "gsm8k": {}}


def dump():
    with open(RESULTS_PATH, "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)


def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)


RESULTS["meta"] = {
    "model": MODEL_ID,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda": torch.version.cuda,
}
log("meta:", RESULTS["meta"])

# ---------------------------------------------------------------- model load
log("downloading model snapshot ...")
t0 = time.time()
local = snapshot_download(MODEL_ID)
log(f"snapshot ready in {time.time()-t0:.0f}s -> {local}")

tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
eos = tok.eos_token_id
log("eos_token_id:", eos, "| mask_token_id:", 100)

def _load(dtype):
    m = AutoModel.from_pretrained(local, trust_remote_code=True).cuda().to(dtype).eval()
    return m

def _degenerate(text):
    t = text.strip()
    if len(t) < 5:
        return True
    ws = t.split()
    if len(ws) >= 20 and len(set(ws[-20:])) <= 2:
        return True
    return False

dtype = torch.float16
model = _load(dtype)
log("model loaded in", dtype)


def chat_ids(text):
    messages = [{"role": "user", "content": text}]
    p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(p, return_tensors="pt").input_ids.cuda()


GEN_KW = dict(max_new_tokens=256, block_length=32, threshold=0.9)

# ---------------------------------------------------------------- 1. sanity
SANITY = [
    "The capital of France is",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
]
for i, q in enumerate(SANITY):
    ids = chat_ids(q)
    out, nfe = model.generate(ids, eos_token_id=eos, **GEN_KW)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    log(f"sanity[{i}] nfe={nfe} out={text[:200]!r}")
    RESULTS["sanity"][f"prompt{i}"] = dict(nfe=nfe, text=text[:1000])
    if _degenerate(text) and dtype == torch.float16:
        log("degenerate fp16 output -> reloading in fp32")
        del model
        torch.cuda.empty_cache()
        dtype = torch.float32
        model = _load(dtype)
        out, nfe = model.generate(ids, eos_token_id=eos, **GEN_KW)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        log(f"sanity[{i}] fp32 nfe={nfe} out={text[:200]!r}")
        RESULTS["sanity"][f"prompt{i}_fp32"] = dict(nfe=nfe, text=text[:1000])
RESULTS["meta"]["dtype_final"] = str(dtype)
dump()

# ---------------------------------------------------------------- 2. equivalence
log("equivalence check: stock vs wet_ink(revision_blocks=0) for every policy")
eq = {"ids_equal": [], "nfe_equal": [], "nfes_stock": [], "nfes_w0": {}}
for i, q in enumerate(SANITY):
    ids = chat_ids(q)
    o_s, n_s = model.generate(ids, eos_token_id=eos, **GEN_KW)
    eq["ids_equal"].append(True)
    eq["nfe_equal"].append(True)
    eq["nfes_stock"].append(int(n_s))
    eq["nfes_w0"][f"prompt{i}"] = {}
    for pol in ("reme_di", "codr", "per_step"):
        o_w, n_w = generate_wet_ink(
            model, ids, revision_blocks=0, remask_policy=pol,
            eos_token_id=eos, **GEN_KW
        )
        ok_ids = bool(torch.equal(o_s, o_w))
        ok_nfe = n_s == n_w
        eq["nfes_w0"][f"prompt{i}"][pol] = dict(
            ids_equal=ok_ids, nfe_equal=ok_nfe, nfe=int(n_w)
        )
        if not ok_ids or not ok_nfe:
            eq["ids_equal"][-1] = False
            eq["nfe_equal"][-1] = False
        log(f"eq[{i}] policy={pol}: ids_equal={ok_ids} nfe {n_s} vs {n_w}")
RESULTS["equivalence"] = eq
dump()

# ---------------------------------------------------------------- 3. qualitative
QUAL = [
    "Write a short paragraph (3 sentences) explaining why the sky is blue.",
    "A farmer has 17 sheep. All but 9 run away. How many sheep does he have left? Explain briefly.",
    "Translate into English and continue naturally: 'Le petit chat dort sur'",
]
for q in QUAL:
    ids = chat_ids(q)
    entry = {"question": q}
    o_s, n_s = model.generate(ids, eos_token_id=eos, **GEN_KW)
    entry["stock"] = dict(
        text=tok.decode(o_s[0, ids.shape[1]:], skip_special_tokens=True)[:1500], nfe=int(n_s)
    )
    for tag, kw in (
        ("reme_di_s32", dict(remask_policy="reme_di")),
        ("codr", dict(remask_policy="codr")),
    ):
        o_w, n_w, st = generate_wet_ink(
            model, ids, revision_blocks=1, eos_token_id=eos, return_stats=True,
            **{**GEN_KW, **kw},
        )
        entry[tag] = dict(
            text=tok.decode(o_w[0, ids.shape[1]:], skip_special_tokens=True)[:1500],
            nfe=int(n_w),
            remask_events=st["remask_events"],
            revision_token_changes=st["revision_token_changes"],
            blocks_with_remask=st["blocks_with_remask"],
            codr_drift_remasks=st["codr_drift_remasks"],
            per_block=st["per_block"][:6],
        )
    entry["n_tokens_changed_vs_stock"] = int(
        (o_s[0, ids.shape[1]:] != o_w[0, ids.shape[1]:]).sum().item()
    ) if o_s.shape == o_w.shape else None
    log(f"qual: {q[:40]!r} | stock nfe={n_s}")
    RESULTS["qualitative"].append(entry)
    dump()

# ---------------------------------------------------------------- 4. GSM8K A/B
# Protocol mirrors the official evaluate.py (NVlabs repo): instruction, chat
# template, last-\boxed{}-or-last-number scoring, NFE accounting.
INSTRUCTION = (
    "Solve the following math problem. Put the final numerical answer "
    "inside \\boxed{} at the very end.\n\n"
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def _last_number(text):
    cleaned = text.replace(",", "")
    matches = _NUMBER_RE.findall(cleaned)
    return matches[-1] if matches else None


def _boxed_answer(text):
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _numbers_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a.strip() == b.strip()


def score_gsm8k(model_out, gold):
    pred = _boxed_answer(model_out) or _last_number(model_out)
    return pred is not None and _numbers_equal(pred, gold)


def get_ds():
    for repo in ("gsm8k", "openai/gsm8k"):
        try:
            return load_dataset(repo, "main", split="test")
        except Exception as e:  # noqa: BLE001
            log(f"load_dataset({repo}) failed: {e}")
    raise RuntimeError("cannot load GSM8K")


def run_config(name, mode, limit=50, max_new_tokens=512, block_length=32,
               threshold=0.9, revision_blocks=1, remask_policy="reme_di",
               criterion="confidence", denoise_steps=None, drift_threshold=0.0):
    if time.time() - T0 > TIME_LIMIT_S:
        log(f"SKIP {name}: time budget exhausted")
        RESULTS["gsm8k"][name] = {"skipped": "time budget"}
        dump()
        return
    ds = get_ds()
    n = min(limit, len(ds))
    log(f"config {name}: mode={mode} limit={n} max_new={max_new_tokens} "
        f"block={block_length} policy={remask_policy} crit={criterion} "
        f"steps={denoise_steps} thr={threshold} W={revision_blocks if mode=='wet' else 0}")
    t0 = time.time()
    correct = 0
    tot_tok = 0
    tot_nfe = 0
    agg = dict(remask_events=0, revision_token_changes=0, blocks_with_remask=0,
               safety_tail_steps=0, hard_cap_hits=0, codr_probe_forwards=0,
               codr_drift_remasks=0, codr_refill_forwards=0)
    details = []
    for i in range(n):
        row = ds[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        ids = chat_ids(INSTRUCTION + row["question"])
        try:
            if mode == "stock":
                out, nfe = model.generate(
                    ids, max_new_tokens=max_new_tokens, block_length=block_length,
                    threshold=threshold, eos_token_id=eos, max_thinking_tokens=6000,
                )
                st = None
            else:
                out, nfe, st = generate_wet_ink(
                    model, ids, max_new_tokens=max_new_tokens, block_length=block_length,
                    revision_blocks=revision_blocks, remask_policy=remask_policy,
                    criterion=criterion, denoise_steps=denoise_steps,
                    threshold=threshold, drift_threshold=drift_threshold,
                    eos_token_id=eos, max_thinking_tokens=6000, return_stats=True,
                )
            new = out[0, ids.shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            ok = score_gsm8k(text, gold)
            correct += int(ok)
            tot_tok += int(new.numel())
            tot_nfe += int(nfe)
            if st is not None:
                agg["remask_events"] += st["remask_events"]
                agg["revision_token_changes"] += st["revision_token_changes"]
                agg["blocks_with_remask"] += st["blocks_with_remask"]
                agg["safety_tail_steps"] += st["safety_tail_steps"]
                agg["hard_cap_hits"] += st.get("hard_cap_hits", 0)
                agg["codr_probe_forwards"] += st["codr_probe_forwards"]
                agg["codr_drift_remasks"] += st["codr_drift_remasks"]
                agg["codr_refill_forwards"] += st["codr_refill_forwards"]
            if i < 5 or (i + 1) % 10 == 0:
                log(f"  {name}[{i+1}/{n}] ok={int(ok)} tok={int(new.numel())} nfe={nfe} acc={100.0*correct/(i+1):.1f}%")
            details.append(dict(i=i, ok=bool(ok), n_tokens=int(new.numel()), nfe=int(nfe),
                                text=text[:400]))
        except torch.cuda.OutOfMemoryError:
            log(f"  {name}[{i}] OOM -> skipping problem")
            torch.cuda.empty_cache()
            details.append(dict(i=i, error="oom"))
    total = len([d for d in details if "error" not in d])
    res = dict(
        mode=mode, revision_blocks=revision_blocks if mode == "wet" else 0,
        remask_policy=remask_policy if mode == "wet" else "stock",
        criterion=criterion, denoise_steps=denoise_steps,
        block_length=block_length, threshold=threshold, limit=n,
        accuracy=100.0 * correct / max(total, 1),
        avg_tokens=tot_tok / max(total, 1),
        avg_nfe=tot_nfe / max(total, 1),
        tpf=tot_tok / max(tot_nfe, 1),
        wall_seconds=time.time() - t0,
        **agg,
        details=details,
    )
    RESULTS["gsm8k"][name] = res
    log(f"== {name}: acc={res['accuracy']:.2f}% avg_tok={res['avg_tokens']:.1f} "
        f"avg_nfe={res['avg_nfe']:.1f} TPF={res['tpf']:.2f} wall={res['wall_seconds']:.0f}s "
        f"remask={agg['remask_events']} revised={agg['revision_token_changes']} "
        f"drift={agg['codr_drift_remasks']}")
    dump()


run_config("stock_b32_t0.9", "stock", threshold=0.9)
run_config("remedi_w1_s32_conf", "wet", remask_policy="reme_di", denoise_steps=32)
run_config("remedi_w1_s16_conf", "wet", remask_policy="reme_di", denoise_steps=16)
run_config("remedi_w1_s16_entropy", "wet", remask_policy="reme_di",
           criterion="entropy", denoise_steps=16)
run_config("codr_w1_t0.9", "wet", remask_policy="codr", threshold=0.9)
run_config("remedi_w2_s16_conf", "wet", remask_policy="reme_di",
           denoise_steps=16, revision_blocks=2)

# ---------------------------------------------------------------- summary
log("=========== SUMMARY ===========")
log(f"{'config':<22}{'acc%':>7}{'avg_tok':>9}{'avg_nfe':>9}{'TPF':>7}{'remask':>8}{'revised':>9}{'drift':>7}{'wall_s':>8}")
for name, r in RESULTS["gsm8k"].items():
    if "skipped" in r:
        log(f"{name:<22} SKIPPED")
        continue
    log(f"{name:<22}{r['accuracy']:>7.2f}{r['avg_tokens']:>9.1f}{r['avg_nfe']:>9.1f}"
        f"{r['tpf']:>7.2f}{r['remask_events']:>8}{r['revision_token_changes']:>9}"
        f"{r['codr_drift_remasks']:>7}{r['wall_seconds']:>8.0f}")
dump()
log("DONE. results.json written to", RESULTS_PATH)
