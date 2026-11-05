"""
wet_ink.py — training-free "wet-ink" decoding for Nemotron-Labs-Diffusion.

Concept (user's idea)
---------------------
The diffusion frontier opens tokens left-to-right. Already-opened ("committed")
tokens inside a bounded revision window behind the frontier are re-examined
while the frontier writes the next block: if the model's confidence in a
committed token drops below the threshold, the token returns to [MASK] and is
re-predicted together with the still-wet current block. Ink near the frontier
stays correctable; ink far from the frontier dries (leaves the window).

With revision_blocks=0 this function is behaviourally IDENTICAL to the stock
``NemotronLabsDiffusionModel.generate()`` (bit-identical at temperature 0) —
verified by tests/test_smoke.py.

Provenance of every mechanism (NO new formulas, NO modified formulas)
---------------------------------------------------------------------
All references are to the files of nvidia/Nemotron-Labs-Diffusion-3B
(``modeling_nemotron_labs_diffusion.py``, shipped on HF, sha-matched to the
public repo) and to the official LLaDA repository (ML-GSAI/LLaMA,
``generate.py``). The model's own module-level helpers are imported FROM the
loaded model at runtime — we never re-implement them.

1. Commit rule — stock ``_get_transfer_index`` (modeling.py L835–861), called
   verbatim: Gumbel-argmax x0 (L842–843), confidence x0_p = softmax(logits)[x0]
   (L845–846, identical formula to LLaDA generate.py L180–183 and to
   ``sampler.py::_unmask_low_confidence`` L10–14 of the NVlabs repo), top-k
   budget + optional threshold filter with the top-1 always committed
   (L852–860). Budget split: stock ``_get_num_transfer_tokens`` (L824–832),
   identical to LLaDA ``get_num_transfer_tokens`` (generate.py L22–40).

2. Re-masking rule (the ONLY addition) — "LLaDA low-confidence remasking"
   applied to OPEN positions of the revision window: a second verbatim call to
   the stock ``_get_transfer_index`` with ``mask_index = open window
   positions``. Its threshold filter implements the model's own confidence
   test ``draft_conf >= threshold`` (modeling.py L727–736,
   linear_spec_generate); positions that FAIL the test (complement of the
   returned transfer_index) return to [MASK]. Vanilla LLaDA applies this
   confidence formula only to masked positions (its generate.py keeps
   ``confidence = torch.where(mask_index, x0_p, -inf)``); extending the same
   published formula + the same published threshold test to open positions of
   the window is exactly the approved "Комбинация №1" (LLaDA remasking x
   Nemotron generator). State-space validity: training masks i.i.d. Bernoulli
   over all positions (modeling.py L220–232), so any mask subset is an
   in-distribution state.

3. KV-cache surgery — stock ``_crop_dynamic_cache`` (modeling.py L803–811),
   the same helper ``linear_spec_generate`` uses to roll the cache back on
   rejected speculation (L775). The revision span is re-denoised
   bidirectionally against the cropped cache via the stock forward path
   (``use_cache=False`` concatenates cache K/V without writing, exactly as in
   stock block denoising, modeling_ministral.py L159–162 + L168–179), then the
   cache is rebuilt causally by the stock post-block forward
   (``use_cache=True, use_causal_mask=True``, modeling.py L500–519).

4. Span alignment — the revision window is a whole number of blocks of
   ``block_size`` (config block_size=32), matching the block structure the
   model was trained with; the noisy span attends to the clean prefix exactly
   like a training-time noisy block (M_OBC pattern).

Everything else below is control flow around these stock primitives.
"""

import importlib
from typing import Optional

import torch


def get_stock_helpers(model):
    """Import the module that defines `type(model)` (the remote-code module
    loaded by transformers from the HF repo) and return the model's OWN helper
    functions, so every formula is executed from the original source file."""
    module = importlib.import_module(type(model).__module__)
    return (
        module._get_transfer_index,
        module._get_num_transfer_tokens,
        module._crop_dynamic_cache,
    )


@torch.no_grad()
def generate_wet_ink(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    block_length: int,
    revision_blocks: int = 1,
    threshold: Optional[float] = 0.9,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    causal_context: bool = True,
    remasking: bool = True,
    commit: str = "threshold",   # "threshold" (stock) | "fixed1" (sampler.py 'fixed', k=1)
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    return_stats: bool = False,
):
    """Stock block-diffusion ``generate()`` + bounded revision window.

    Differences from stock (modeling.py generate, L379–529) are marked
    ``# WET-INK:`` — everything else is a verbatim structural copy:

      * the denoised span is [last `revision_blocks` completed blocks] +
        [current block] instead of just the current block;
      * the KV cache is cropped to the span start (stock helper) before the
        block and rebuilt causally over the whole span afterwards;
      * open tokens of the revision window whose confidence dropped below
        `threshold` return to [MASK] (LLaDA low-confidence remasking) while
        the frontier is still writing the current block.

    Returns (output_ids, nfe) like stock; with return_stats=True also a stats
    dict. NFE is counted exactly like stock generate() (denoise steps +
    post-block forwards; prefill is not counted, matching stock L414/L476/L511).
    """
    _get_transfer_index, _get_num_transfer_tokens, _crop_dynamic_cache = get_stock_helpers(model)

    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    mask_id = model.mask_token_id

    x_accum = prompt_ids.clone()
    B = prompt_ids.shape[0]
    prompt_len = prompt_ids.size(1)

    assert max_new_tokens % block_length == 0
    num_blocks = max_new_tokens // block_length

    nfe = 0
    stats = {
        "denoise_forwards": 0,
        "post_block_forwards": 0,
        "commit_events": 0,
        "remask_events": 0,
        "blocks_with_remask": 0,
        "revision_token_changes": 0,
        "safety_tail_steps": 0,
        "per_block": [],
    }

    def _set_diffusion_lm(val: bool):
        for layer in model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = val

    # Initial causal prefill produces the KV cache and the next-block seed.
    # [stock L421–435, verbatim]
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
        # WET-INK: span = [revision window (W previously completed blocks,
        # block-aligned) + current block]. With revision_blocks=0 the span
        # degenerates to the stock block.
        block_start = prompt_len + num_block * block_length
        block_slice = slice(block_start, block_start + block_length)
        rev_start = prompt_len + max(0, num_block - revision_blocks) * block_length
        span_slice = slice(rev_start, block_start + block_length)
        span_len = span_slice.stop - span_slice.start
        window_len = block_start - rev_start  # 0 => pure stock behaviour

        mask_block = torch.full(
            (B, block_length), mask_id, dtype=prompt_ids.dtype, device=prompt_ids.device,
        )
        if causal_context:
            mask_block[:, 0] = next_token[:, 0]

        x_accum = torch.cat([x_accum, mask_block], dim=1)

        # Thinking-budget enforcement. [stock L448–465, verbatim]
        if end_think_token_id is not None and max_thinking_tokens is not None:
            tokens_before = num_block * block_length
            tokens_after = tokens_before + block_length
            if tokens_after > max_thinking_tokens:
                gen_so_far = x_accum[:, prompt_len:block_start]
                has_end_think = (
                    (gen_so_far == end_think_token_id).any(dim=1)
                    if gen_so_far.size(1) > 0
                    else torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
                )
                if not has_end_think.all():
                    offset = max(0, max_thinking_tokens - tokens_before)
                    inject_pos = block_start + offset
                    for b in range(B):
                        if not has_end_think[b]:
                            x_accum[b, inject_pos] = end_think_token_id

        mask_span_idx0 = x_accum[:, span_slice] == mask_id
        steps_cap = span_len  # one denoising step per span token (stock rule, extended to |S|)
        num_transfer_tokens = _get_num_transfer_tokens(mask_span_idx0, steps_cap)
        commit_threshold = threshold if commit == "threshold" else None
        # 'fixed1' = sampler.py 'fixed' strategy with budget k=1 (NVlabs repo,
        # xp/dlm_api/dlm_generate/utils/sampler.py): exactly one token committed
        # per step, chosen by confidence.

        def _budget_for_step(mask_idx_now, i):
            """Per-step commit budget. threshold mode: stock schedule (the value
            is overridden inside _get_transfer_index anyway). fixed1: ones.
            Budget mode (threshold=None, not fixed1): re-apply the SAME stock
            split (_get_num_transfer_tokens) to the CURRENT mask count over the
            remaining steps — identical to the stock schedule when no remasking
            happens, and still >= 1 whenever masks remain after re-masking."""
            if commit == "fixed1":
                return torch.ones(B, dtype=torch.long, device=prompt_ids.device)
            if commit_threshold is not None:
                return num_transfer_tokens[:, min(i, steps_cap - 1)]
            remaining = max(1, steps_cap - i)
            return _get_num_transfer_tokens(mask_idx_now, remaining)[:, 0]

        # Bookkeeping: window tokens at block entry (to measure real revisions).
        window_entry = (
            x_accum[:, rev_start:block_start].clone() if window_len > 0 else None
        )

        # WET-INK: crop the KV cache to the start of the revision span.
        # Stock helper, same role as in linear_spec_generate L775.
        if window_len > 0:
            _crop_dynamic_cache(past_key_values, rev_start)

        remasked_in_block = 0
        stopped_by_eos = False
        hard_cap = 4 * steps_cap  # control-flow safety bound only (never reached
        #                                  in stock-equivalent mode; counted in stats)

        # ---- denoise loop over the span [stock L471–498, span-extended] ----
        for i in range(hard_cap):
            span_ids = x_accum[:, span_slice]
            mask_span_idx = span_ids == mask_id
            if mask_span_idx.sum() == 0:
                break

            nfe += 1
            stats["denoise_forwards"] += 1
            logits_span = model(
                span_ids,
                past_key_values=past_key_values,
                use_cache=False,
            ).logits

            budget_i = _budget_for_step(mask_span_idx, i)
            x0, transfer_idx = _get_transfer_index(
                logits_span, temperature, mask_span_idx, span_ids,
                num_transfer_tokens=budget_i, threshold=commit_threshold,
            )
            cur = span_ids.clone()
            cur[transfer_idx] = x0[transfer_idx]
            x_accum[:, span_slice] = cur
            stats["commit_events"] += int(transfer_idx.sum().item())

            # ---- WET-INK: LLaDA low-confidence remasking of OPEN window tokens ----
            # While the frontier is still writing the current block, re-examine
            # the committed tokens of the revision window with the SAME stock
            # formula and threshold: open positions failing the confidence test
            # return to [MASK] and rejoin the denoising pool.
            # (Requires threshold is not None — the criterion IS the threshold.)
            if (
                remasking
                and window_len > 0
                and threshold is not None
            ):
                new_block_mask_idx = x_accum[:, block_slice] == mask_id
                if bool(new_block_mask_idx.any()):
                    win_ids = x_accum[:, rev_start:block_start]
                    win_open = win_ids != mask_id
                    if bool(win_open.any()):
                        _, keep_idx = _get_transfer_index(
                            logits_span[:, :window_len], temperature, win_open, win_ids,
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

            # EOS check. [stock L491–498, verbatim on the current block]
            if eos_token_id is not None:
                block_tokens = x_accum[:, block_slice]
                eos_mask = block_tokens == eos_token_id
                if eos_mask.any(dim=1).any():
                    after_eos = eos_mask.cumsum(dim=1).bool()
                    mask_before = (block_tokens == mask_id) & ~after_eos
                    if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                        stopped_by_eos = True
                        break

        if i == hard_cap - 1 and not stopped_by_eos:
            stats["hard_cap_hits"] = stats.get("hard_cap_hits", 0) + 1

        # WET-INK (safety tail, control flow only): guarantee the span is fully
        # denoised before the causal cache rebuild — the stock invariant "no
        # [MASK] enters the KV cache" (stock guarantees it via forced top-1
        # commits over exactly `steps` steps; re-masking can add masks, so we
        # run commit-only steps until clear). Skipped entirely when the stock
        # EOS early-stop fired: stock itself proceeds to the post-block with
        # masks still present (the run ends right after, so the cache is never
        # reused) — we keep byte-for-byte stock semantics in that path.
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
                    span_ids, past_key_values=past_key_values, use_cache=False
                ).logits
                budget_i = (
                    torch.ones(B, dtype=torch.long, device=prompt_ids.device)
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
                if eos_token_id is not None:
                    block_tokens = x_accum[:, block_slice]
                    eos_mask = block_tokens == eos_token_id
                    if eos_mask.any(dim=1).any():
                        after_eos = eos_mask.cumsum(dim=1).bool()
                        mask_before = (block_tokens == mask_id) & ~after_eos
                        if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                            break

        # Post-block: causal forward over the WHOLE span to rebuild the KV
        # cache and (when causal_context) sample the seed for the next block.
        # [stock L500–519; span-extended so revised tokens re-enter the cache]
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

        # EOS final check. [stock L521–527, verbatim]
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
