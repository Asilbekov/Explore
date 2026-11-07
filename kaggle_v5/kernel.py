#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wet-Ink v5 — OUR METHOD, implemented: sliding-window frontier decoder on
LLaDA2.2-mini's own formulas (2×Tesla T4, 4-bit NF4, one model per GPU).

WHAT CHANGED vs v4 (owner's correction, 7 Nov 2026): v4 only ablated the
model's stock generate() parameters. That measured the base, not our idea.
v5 implements the idea from the report §2.1 as a real decoder:

  (1) FRONTIER: open EXACTLY ONE token per diffusion step, left-to-right
      (AR commit of the leftmost mask). This is the left-to-right end of the
      published interpolation spectrum: BD3-LM (Arriola et al., ICLR 2025
      Oral) and Set Diffusion (Arriola et al. 2026) treat tokens-per-step /
      position orderings as free inference-time variables.
  (2) WET WINDOW: the model's OWN trained T2T gate (LLaDA2.2 L-EBPO;
      LLaDA2.1 "Speeding Up Text Diffusion via Token Editing",
      arXiv:2602.08676) — p0 > editing_threshold & old != x0 -> greedy
      rewrite — restricted to the sliding window [f-W, f) behind the
      frontier. Hard dry edge: "ink dries with distance" (the idea's core).
  (3) Trained edit ops SPLIT/DELETE, anti-loop resampler, M2T/T2T write
      semantics, block-diagonal attention formula — ALL reused verbatim
      from the model (Levenshtein-Transformer lineage, Gu et al. 2019;
      trained into LLaDA2.2 via L-EBPO).

THE FOUR RULES (report §2.2, owner's clarification of 7 Nov 2026):
  (a) study the base model's formulas from primary sources — done,
      modeling_llada2_moe.py read verbatim (1 801 lines);
  (b) no retraining — inference-time only;
  (c) pick the MOST EFFECTIVE formulas: the trained T2T gate over any
      training-free gate (our v3: training-free remasking hurt, -12.5..-26
      pp; corroborated by "Re-evaluating Confidence Remasking",
      arXiv:2606.12232 — flip-flop behaviour);
  (d) no self-invented formulas; combining published formulas (and their
      surrounding code) from any architectures IS allowed after a
      compatibility check. The compatibility argument: every component
      operates on the model's own trained primitives; the ONLY new thing is
      the decoding SCHEDULE (which position opens / which window stays
      editable per step) — exactly the free variable identified by
      BD3-LM / Set Diffusion. THE GOAL: realize the owner's idea, in the
      most effective published way.

ARMS (v5): wetink W∈{8,16,32} (T2T ON), wetink_noedit_w16 (T2T OFF —
frontier-only control), he_wetink_w16 (HumanEval). Baselines stock_b32 /
noedit_b32 / edit0.5/0.7 come from the completed v4b run (same model, same
4-bit quant, same prompts, same gen_length=160) — paired McNemar offline.

Expected cost (honest): NFE ≈ one forward per committed token ≈ 160-200 per
task vs stock's ~39 — the price of the one-token frontier (~4-5×). Reported
as NFE ratio in every comparison.

Output: /kaggle/working/results_v5.json
"""

import base64
import gzip
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

# Mitigate CUDA fragmentation (push-2 OOM'd on GPU 1 with 9 GiB "reserved but
# unallocated" — classic fragmentation pattern in materialize_copy).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _pip(*pkgs):
    print("pip install:", *pkgs, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


# transformers 5.2.0 is REQUIRED by LLaDA2.2-mini (config.json: transformers_version=5.2.0)
# bitsandbytes for int8 quantization on T4; accelerate for device_map="auto" (2×T4)
# bnb >= 0.48.0 ships the libbitsandbytes_cuda130.so binary that Kaggle (CUDA 13) needs;
# earlier bnb 0.46.x only had cuda12x — that was the v4 push-1 failure.
_pip("transformers==5.2.0", "datasets", "accelerate", "bitsandbytes>=0.48.2",
     "sentencepiece", "protobuf")

import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.utils.loading_report as _lr  # noqa: E402
import transformers.modeling_utils as _mu  # noqa: E402

# ---- bnb 0.46+ renamed Int8Params.SCB -> CB; transformers 5.2.0's loading_report
#      still calls .SCB and raises AttributeError+RuntimeError at finalize time,
#      AFTER the model has actually loaded. Patch log_state_dict_report to swallow
#      those reporting-only errors so from_pretrained returns the loaded model.
#      (Quantization itself succeeds — this is a post-load reporting bug only.)
_orig_log_state_dict_report = _lr.log_state_dict_report


def _safe_log_state_dict_report(*a, **kw):
    try:
        return _orig_log_state_dict_report(*a, **kw)
    except (AttributeError, RuntimeError) as e:  # noqa: BLE001
        # Most common: "'Int8Params' object has no attribute 'SCB'. Did you mean: 'CB'?"
        print(f"[loading_report patched] swallowed non-fatal: {type(e).__name__}: {e}",
              flush=True)


_lr.log_state_dict_report = _safe_log_state_dict_report
_mu.log_state_dict_report = _safe_log_state_dict_report

from datasets import load_dataset  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

import bitsandbytes as _bnb  # noqa: E402
print("bitsandbytes version:", _bnb.__version__, flush=True)

MODEL_ID = "inclusionAI/LLaDA2.2-mini"
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS_PATH = os.path.join(OUT_DIR, "results_v5.json")

MASK_ID = 156895
DELETE_ID = 156930
SPLIT_ID = 156931
EOS_ID = 156892

# Kaggle session limit is 9h; keep a 1.5h safety margin
TIME_LIMIT_S = 7.5 * 3600
T0 = time.time()

RESULTS = {
    "meta": {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "quantization": "int4 NF4 (bitsandbytes, double-quant, fp16 compute)",
        "attn_impl": "eager",
        "suite": "v5: wet-ink sliding-window frontier decoder (OUR idea §2.1) "
                 "on LLaDA2.2's own formulas; baselines from v4b",
        "idea": "frontier opens ONE token/step left-to-right; trailing window "
                "W stays editable via the model's trained T2T gate; hard dry "
                "edge at distance W (ink dries)",
        "components": {
            "frontier": "AR commit of leftmost mask (BD3-LM / Set Diffusion "
                        "spectrum end; our schedule)",
            "revision": "LLaDA2.2 trained T2T gate p0>editing_threshold & "
                        "old!=x0 (L-EBPO), windowed to [f-W, f)",
            "edit_ops": "SPLIT/DELETE via model's own "
                        "_apply_edit_operations_with_tracking (LevT lineage)",
            "anti_loop": "model's own _resample_to_escape_loop",
            "attention": "model's own block-diagonal formula (verbatim)",
        },
        "v4b_baselines": "stock_b32 / noedit_b32 / edit0.5/0.7 from the "
                         "completed v4b run (results_v4.json); paired offline",
        "rules": "no self-invented formulas; combining published formulas + "
                 "code from any architectures allowed after compatibility "
                 "check; goal = implement the owner's wet-ink idea most "
                 "effectively (rules clarified 7 Nov 2026)",
        "dropped_from_v3": ["RemeDi (training-free remasking: harmful, v3)",
                            "CoDR (training-free drift pass: neutral, v3)"],
    },
    "sanity": {},
    "equivalence": {},
    "gsm8k": {},
    "humaneval": {},
    "stats": {},
}
EST = {}
LOCK = threading.RLock()        # guards RESULTS + EST (two worker threads)
ACTIVE = {"n": 0}               # workers currently consuming the wall clock
ACT_LOCK = threading.Lock()


def dump():
    with LOCK:
        tmp = RESULTS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(RESULTS, f, indent=1)
        os.replace(tmp, RESULTS_PATH)


def log(*a):
    el = time.time() - T0
    h, m = int(el // 3600), int((el % 3600) // 60)
    print(f"[{h:02d}:{m:02d}]", *a, flush=True)


def remaining():
    return TIME_LIMIT_S - (time.time() - T0)


# ---------------------------------------------------------------- model load
log("downloading model snapshot ...")
t0 = time.time()
local = snapshot_download(MODEL_ID)
log(f"snapshot ready in {time.time()-t0:.0f}s -> {local}")

tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
log("tokenizer ready; eos:", tok.eos_token_id, "| mask:", MASK_ID,
    "| pad:", tok.pad_token_id)


def _load(gpu):
    """Load LLaDA2.2-mini in 4-bit NF4 onto ONE T4 (whole model, no split).

    - Why 4-bit (v4b): the int8 run (v4a) needed both GPUs (~17 GiB weights),
      forcing a 2-GPU pipeline; with bnb's int8 matmul path on SM75 this gave
      ~3.8-4.2 s per forward (~140-220 s per GSM8K task) and the 7.5 h budget
      covered only 2 of 14 arms. NF4 storage for the 16.3B MoE is ~8.1 GiB
      (quantized Linears: 19 MoE layers x 256 experts + attn) + fp16
      lm_head/word_embeddings (~0.6 GiB each, 157k vocab) + fp32 router ≈
      ~9.4 GiB total — a FULL copy fits on one 15.8 GiB T4 with ~6 GiB
      headroom for activations (use_cache=False: diffusion full-forwards).
    - bnb_4bit_compute_dtype=float16: T4 (SM75) has no bf16 tensor cores.
    - bnb_4bit_use_double_quant=True: saves ~0.4 GiB on a 16B checkpoint.
    - llm_int8_skip_modules=["lm_head"]: 157k-vocab output head stays fp16
      (transformers' own default for bnb; kept explicit).
    - attn_implementation='eager': required — the model builds its block
      attention mask in bfloat16, and SDPA rejects a bf16 mask that does
      not match the query dtype. eager attention promotes the mask to fp32
      via softmax(... dtype=torch.float32), so it accepts any mask dtype.
    - MoE experts are the model's own LLaDA2MoeMLP — ordinary nn.Linear,
      bnb-compatible; the router is a raw nn.Parameter (router_dtype fp32),
      so bnb leaves it unquantized automatically.
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (env, set above)
      still defeats the fragmentation pattern from the v4 push-2 OOM.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"],
    )
    m = AutoModelForCausalLM.from_pretrained(
        local,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map={"": gpu},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    m.eval()
    m._nfe_state = dict(nfe=0, prev=None, churn=0, special_residual=0)
    return m


def _attach_meter(model):
    """Observation-only per-instance NFE/churn counter (does NOT alter
    generation math). Each worker thread drives its own model instance,
    so the meter state lives on the model object itself."""
    st = model._nfe_state
    _orig_forward = model.forward

    def _wrapped_forward(*a, **k):
        st["nfe"] += 1
        ids = a[0] if a else k.get("input_ids")
        if ids is not None:
            cur = ids.detach()
            prev = st["prev"]
            if prev is not None and prev.shape == cur.shape:
                changed = (cur != prev) & (prev != MASK_ID)
                st["churn"] += int(changed.sum().item())
            st["prev"] = cur.clone()
        return _orig_forward(*a, **k)

    model.forward = _wrapped_forward


log("loading model #0 in 4-bit NF4 on GPU0 (this takes ~5-10 min) ...")
t0 = time.time()
M0 = _load(0)
_attach_meter(M0)
log(f"model #0 loaded in {time.time()-t0:.0f}s")
RESULTS["meta"]["device_map"] = str(getattr(M0, "hf_device_map", "single"))
dump()

# ---------------------------------------------------------------- prompt builders
def chat_ids(text, dev):
    messages = [{"role": "user", "content": text}]
    p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(p, return_tensors="pt").input_ids.to(dev)


# ---------------------------------------------------------------- sanity + equivalence
SANITY = ("Natalia sold clips to 48 of her friends in April, and then she sold "
          "half as many clips in May. How many clips did Natalia sell altogether "
          "in April and May?")


def _degenerate(text):
    t = text.strip()
    if len(t) < 5:
        return True
    ws = t.split()
    if len(ws) >= 20 and len(set(ws[-20:])) <= 2:
        return True
    return False


# ================================================================ OUR DECODER
def wetink_generate(model, ids, gen_length=160, block_length=32, window=16,
                    editing_threshold=0.0, max_post_steps=4, temperature=0.0,
                    eos_early_stop=True):
    """THE WET-INK METHOD (report §2.1), built on the model's own formulas.

    Schedule (OUR idea — the only new thing, per BD3-LM/Set Diffusion the
    schedule is a legitimate free variable, not a formula change):
      * FRONTIER: each step opens EXACTLY ONE token — the LEFTMOST remaining
        mask of the current block — committing it AR-style (sampled x_s,
        greedy at temperature=0). Progress is guaranteed: one mask fewer
        per step. This is the one-token-per-step left-to-right frontier of
        the idea; the AR end of the published AR↔diffusion spectrum.
      * WET WINDOW: positions [f-W, f) behind the frontier (clipped to the
        current block) stay revisable through the model's TRAINED T2T gate
        (verbatim formula from _joint_decode_block / L-EBPO):
            p0 > editing_threshold  AND  old != x0   ->   rewrite to x0
        Hard dry edge: outside the window (or in frozen blocks) a token is
        never touched again — "ink dries with distance".
      * Post-phase: once all masks of the block are resolved, up to
        max_post_steps further steps keep the last W positions wet (the
        drying tail), then the block freezes forever (stock's semantics,
        smaller budget).

    Everything else is reused verbatim from the model (rule (d)):
      - block-diagonal attention mask — exact formula from generate();
      - _diffusion_sample — the model's own sampler;
      - _apply_edit_operations_with_tracking — trained SPLIT/DELETE ops;
      - _resample_to_escape_loop — the model's own anti-loop resampler;
      - SPLIT/DELETE suppression on the final round — the model's own
        termination guarantee; EOS tail-cut — the model's own output rule.

    Compatibility check (rule (d)): all writes go through the same gates
    and ops the model was trained with (L-EBPO); the attention pattern is
    the trained blockwise-causal one; only WHICH position opens per step
    (one, leftmost) and WHICH open positions may be rewritten per step
    (window W behind the frontier) differ from stock.

    Returns (output_ids, stats) with stats = {steps, opened, rewrites, resid}.
    """
    device = model.device
    input_ids = ids.to(device)
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # --- attention mask: the model's OWN block-diagonal formula (verbatim) ---
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    bd_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .log()
        .to(torch.bfloat16)
    )
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    x = torch.full((1, total_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.clone()

    stats = dict(steps=0, opened=0, rewrites=0, resid=0)
    prefill_blocks = prompt_length // block_length
    stop = False

    for num_block in range(prefill_blocks, num_blocks):
        if stop:
            break
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        cur_attn = bd_mask[:, :, :block_end, :block_end]
        cur_pos = position_ids[:, :block_end]

        prompt_mask_block = (x[0, block_start:block_end] != MASK_ID).clone()
        is_original_mask = (x[0, block_start:block_end] == MASK_ID).tolist()
        if sum(is_original_mask) == 0:
            continue  # fully prompt/context block

        seen = set()
        post_steps = 0
        step_id = 0
        # one opened token per step + SPLIT insertions + drying tail
        step_cap = 3 * block_length + max_post_steps + 8

        while True:
            blk = x[0, block_start:block_end]
            old_block = blk.clone()
            mask_index = (old_block == MASK_ID) & (~prompt_mask_block)
            n_mask = int(mask_index.sum().item())

            if n_mask == 0 and post_steps > max_post_steps:
                break
            if step_id >= step_cap:
                break  # residual masks (if any) are counted in stats["resid"]

            logits = model.forward(
                x[:, :block_end], attention_mask=cur_attn, position_ids=cur_pos
            ).logits
            block_logits = logits[0, block_start:block_end, :]

            x_s, p_s = model._diffusion_sample(block_logits, temperature, None, None)
            if temperature != 0.0:
                x0, p0 = model._diffusion_sample(block_logits, 0.0, None, None)
            else:
                x0, p0 = x_s, p_s
            neg_inf = torch.full_like(p0, -float("inf"))

            # ---- FRONTIER: open exactly ONE token — the leftmost mask ----
            mt2_index = torch.zeros(block_length, dtype=torch.bool, device=device)
            frontier = block_length
            if n_mask > 0:
                f = int(
                    torch.nonzero(mask_index, as_tuple=True)[0][0].item()
                )
                mt2_index[f] = True
                frontier = f
                stats["opened"] += 1

            # ---- WET WINDOW: trained T2T gate, restricted to [f-W, f) ----
            editable = (~mask_index) & (~prompt_mask_block)
            win = torch.zeros(block_length, dtype=torch.bool, device=device)
            lo = max(0, frontier - window)
            win[lo:frontier] = True
            t2t_scope = editable & win
            editing_confidence = torch.where(t2t_scope, p0, neg_inf)
            t2t_index = (editing_confidence > editing_threshold) & t2t_scope
            t2t_index = t2t_index & (old_block != x0)

            # ---- final round: suppress SPLIT/DELETE so the block can end ----
            final_round = (n_mask == 0) and (post_steps >= max_post_steps)
            if final_round:
                m2t_sd = mt2_index & ((x_s == SPLIT_ID) | (x_s == DELETE_ID))
                if m2t_sd.any():
                    sd_logits = block_logits[m2t_sd].clone()
                    sd_logits[:, SPLIT_ID] = float("-inf")
                    sd_logits[:, DELETE_ID] = float("-inf")
                    new_tokens, _ = model._diffusion_sample(
                        sd_logits, temperature, None, None
                    )
                    x_s[m2t_sd] = new_tokens
                t2t_sd = t2t_index & ((x0 == SPLIT_ID) | (x0 == DELETE_ID))
                if t2t_sd.any():
                    sd_logits = block_logits[t2t_sd].clone()
                    sd_logits[:, SPLIT_ID] = float("-inf")
                    sd_logits[:, DELETE_ID] = float("-inf")
                    new_tokens, _ = model._diffusion_sample(
                        sd_logits, 0.0, None, None
                    )
                    x0[t2t_sd] = new_tokens

            # ---- writes: M2T sampled, T2T greedy (the model's semantics) ----
            if mt2_index.any():
                blk[mt2_index] = x_s[mt2_index]
            if t2t_index.any():
                blk[t2t_index] = x0[t2t_index]
                stats["rewrites"] += int(t2t_index.sum().item())

            # ---- anti-loop: the model's own resampler ----
            if not final_round and (mt2_index.any() or t2t_index.any()):
                model._resample_to_escape_loop(
                    blk, old_block, block_logits, mt2_index, t2t_index, seen,
                    temperature=temperature, top_k=None, top_p=None,
                )
            seen.add(tuple(blk.tolist()))

            # ---- trained SPLIT/DELETE edit ops (block stays fixed length) ----
            is_original_mask_snapshot = list(is_original_mask)
            edited_block, is_original_mask = (
                model._apply_edit_operations_with_tracking(
                    blk.tolist(),
                    old_block.tolist(),
                    is_original_mask_snapshot,
                    MASK_ID,
                    block_length,
                    delete_token_id=DELETE_ID,
                    split_token_id=SPLIT_ID,
                )
            )
            x[0, block_start:block_end] = torch.tensor(
                edited_block, device=device, dtype=x.dtype
            )

            if n_mask == 0:
                post_steps += 1
            else:
                post_steps = 0
            step_id += 1
            stats["steps"] += 1

        # block frozen from here on ("fully dry")
        if (
            eos_early_stop
            and (x[0, prompt_length:block_end] == EOS_ID).any()
        ):
            stop = True

    # ---- output: the model's own tail-cut rule (cut at first EOS) ----
    generated = x[:, : prompt_length + gen_length]
    eos_positions = (generated[0][prompt_length:] == EOS_ID).nonzero(as_tuple=True)[0]
    first_eos = int(eos_positions[0].item()) if len(eos_positions) > 0 else gen_length
    output = generated[:, prompt_length: prompt_length + first_eos + 1]
    stats["resid"] = int(
        ((output == MASK_ID) | (output == DELETE_ID) | (output == SPLIT_ID)).sum().item()
    )
    return output, stats


def gen(model, ids, mode="stock", window=16, gen_length=256, block_length=32,
        threshold=0.5, editing_threshold=0.0, max_post_steps=16, temperature=0.0):
    """Decode one task.

    mode="stock":  the model's OWN generate() — v4b baseline arms.
    mode="wetink": OUR §2.1 decoder (wetink_generate) — one-token frontier
                   + sliding trained-T2T window on the model's formulas.
    """
    _NFE_STATE = model._nfe_state
    _NFE_STATE.update(nfe=0, prev=None, churn=0, special_residual=0,
                      wetink_stats=None)
    if mode == "wetink":
        out, wstats = wetink_generate(
            model, ids, gen_length=gen_length, block_length=block_length,
            window=window, editing_threshold=editing_threshold,
            max_post_steps=max_post_steps, temperature=temperature,
        )
    else:
        out = model.generate(
        inputs=ids,
        temperature=temperature,
        block_length=block_length,
        steps=block_length,
        gen_length=gen_length,
        threshold=threshold,
        editing_threshold=editing_threshold,
        max_post_steps=max_post_steps,
        eos_early_stop=True,
        eos_id=EOS_ID,
        mask_id=MASK_ID,
        delete_token_id=DELETE_ID,
        split_token_id=SPLIT_ID,
        max_steps_per_block=1000,
    )
    nfe = _NFE_STATE["nfe"]
    churn = _NFE_STATE["churn"]
    resid = int(((out == MASK_ID) | (out == DELETE_ID) | (out == SPLIT_ID)).sum().item())
    _NFE_STATE["special_residual"] = resid
    if mode == "wetink":
        _NFE_STATE["wetink_stats"] = wstats
    return out, nfe, churn, resid


ids0 = chat_ids(SANITY, M0.device)
t0 = time.time()
out_s, nfe_s, churn_s, resid_s = gen(M0, ids0, mode="stock", gen_length=96,
                                      block_length=32, threshold=0.5,
                                      editing_threshold=0.0)
text_s = tok.decode(out_s[0], skip_special_tokens=True)
sanity_wall = time.time() - t0
log(f"sanity stock b32: nfe={nfe_s} churn={churn_s} resid={resid_s} "
    f"wall={sanity_wall:.0f}s out={text_s[:200]!r}")
RESULTS["sanity"]["stock_b32"] = dict(
    nfe=int(nfe_s), churn=int(churn_s), special_residual=int(resid_s),
    wall_seconds=sanity_wall, output_prefix=text_s[:300],
)
dump()

log("sanity: OUR wet-ink decoder (W=16, one-token frontier, trained T2T window)")
t0 = time.time()
out_w, nfe_w, churn_w, resid_w = gen(M0, ids0, mode="wetink", window=16,
                                      gen_length=128, block_length=32,
                                      editing_threshold=0.0, max_post_steps=4)
text_w = tok.decode(out_w[0], skip_special_tokens=True)
wet_wall = time.time() - t0
ws = M0._nfe_state.get("wetink_stats") or {}
log(f"sanity wetink W=16: nfe={nfe_w} churn={churn_w} resid={resid_w} "
    f"steps={ws.get('steps')} opened={ws.get('opened')} rewrites={ws.get('rewrites')} "
    f"wall={wet_wall:.0f}s out={text_w[:200]!r}")
RESULTS["sanity"]["wetink_w16"] = dict(
    nfe=int(nfe_w), churn=int(churn_w), special_residual=int(resid_w),
    wall_seconds=wet_wall, wetink=ws, output_prefix=text_w[:300],
)
dump()

log("equivalence: OUR decoder greedy determinism (same config twice -> identical ids)")
for W in (16,):
    o1, n1, c1, _ = gen(M0, ids0, mode="wetink", window=W, gen_length=64,
                        block_length=32, editing_threshold=0.0, max_post_steps=4)
    o2, n2, c2, _ = gen(M0, ids0, mode="wetink", window=W, gen_length=64,
                        block_length=32, editing_threshold=0.0, max_post_steps=4)
    ok = bool(torch.equal(o1, o2)) and n1 == n2
    RESULTS["equivalence"][f"wetink_w{W}"] = dict(
        ids_equal=ok, nfe1=int(n1), nfe2=int(n2),
        churn1=int(c1), churn2=int(c2))
    log(f"eq wetink W={W}: ids_equal={ok} nfe {n1}=={n2} churn {c1}=={c2}")
dump()


# ---------------------------------------------------------------- statistics helpers
def mcnemar_exact(b, c):
    """Two-sided exact binomial (McNemar) test on discordant pairs b, c."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def ok_of(d):
    v = d.get("ok", d.get("passed"))
    return bool(v)


# Per-task wall priors, CALIBRATED from v4b actuals (results_v4.json,
# kernel freedomform/wet-ink-llada22-mini, 4-bit NF4, 2 workers):
#   per_fwd    = stock_b32.wall/n/avg_nfe     = 2.858 s/NFE  (gsm8k)
#   per_fwd_he = he_stock_b32.wall/n/avg_nfe  = 3.295 s/NFE  (humaneval)
#   wetink = per_fwd * 195 NFE/task (one forward per committed frontier token).
# plan_n() recalibrates from each arm's MEASURED wall/total, so these are
# priors only (first-batch planning).
INIT_EST = {
    ("gsm8k", "stock", 32): 121.0,   # measured: 121.4 s/task
    ("gsm8k", "stock", 64): 111.0,   # measured: 111.0 s/task
    ("gsm8k", "stock", 128): 140.0,  # no data (skipped in v4b), prior kept
    ("humaneval", "stock", 32): 49.0,    # measured: 48.6 s/task
    ("humaneval", "stock", 64): 60.0,    # scaled from gsm8k b64 x he ratio
    ("gsm8k", "wetink", 32): 557.0,      # per_fwd * 195 = 2.858 * 195
    ("humaneval", "wetink", 32): 642.0,  # per_fwd_he * 195 = 3.295 * 195
}


def plan_n(bench, mode, block, cap):
    """How many tasks of this arm can the remaining budget afford?

    Uses the measured per-task time of the same mode+block when available,
    else the calibrated prior. The wall clock is consumed by ACTIVE["n"]
    workers in parallel, so the affordable count is divided by the number
    of live workers. Reserves 900 s for paired stats + summary + dump.
    """
    per_task = EST.get((bench, mode, block)) or INIT_EST.get((bench, mode, block), 250.0)
    n_aff = int((remaining() - 900) / (max(ACTIVE["n"], 1) * per_task))
    n_run = max(0, min(cap, n_aff))
    log(f"plan_n[{bench}/{mode}/b{block}]: cap={cap} per_task_est={per_task:.0f}s "
        f"active={ACTIVE['n']} remaining={remaining():.0f}s -> n_run={n_run}")
    return n_run


# ---------------------------------------------------------------- GSM8K
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


_DS_GSM = None
_DS_LOCK = threading.Lock()


def get_gsm8k():
    global _DS_GSM
    with _DS_LOCK:
        if _DS_GSM is None:
            for repo in ("gsm8k", "openai/gsm8k"):
                try:
                    _DS_GSM = load_dataset(repo, "main", split="test")
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"load_dataset({repo}) failed: {e}")
            if _DS_GSM is None:
                raise RuntimeError("cannot load GSM8K")
    return _DS_GSM


def run_gsm8k(name, model, mode="stock", window=16, editing_threshold=0.0,
              block_length=32, threshold=0.5, max_post_steps=16,
              gen_length=160, limit=200):
    if remaining() < 900:
        RESULTS["gsm8k"][name] = {"skipped": "time budget"}
        dump()
        return
    ds = get_gsm8k()
    n = min(limit, len(ds))
    log(f"gsm8k {name}: limit={n} block={block_length} thr={threshold} "
        f"edit_thr={editing_threshold} mps={max_post_steps}")
    t0 = time.time()
    correct = 0
    tot_tok = 0
    tot_nfe = 0
    tot_churn = 0
    tot_resid = 0
    details = []
    for i in range(n):
        if remaining() < 900 and i > 0:
            log(f"  {name}: task-level early stop at {i}/{n} (time budget)")
            break
        row = ds[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        ids = chat_ids(INSTRUCTION + row["question"], model.device)
        try:
            out, nfe, churn, resid = gen(
                model, ids, mode=mode, window=window, gen_length=gen_length,
                block_length=block_length, threshold=threshold,
                editing_threshold=editing_threshold,
                max_post_steps=max_post_steps,
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            ok = score_gsm8k(text, gold)
            if ok:
                correct += 1
            tot_tok += int(out.numel())
            tot_nfe += nfe
            tot_churn += churn
            tot_resid += resid
            if i < 3 or (i + 1) % 20 == 0:
                log(f"  {name}[{i+1}/{n}] ok={int(ok)} nfe={nfe} churn={churn} "
                    f"tok={int(out.numel())} gold={gold}")
            ws = model._nfe_state.get("wetink_stats") or {}
            details.append(dict(i=i, ok=bool(ok), n_tokens=int(out.numel()),
                                nfe=int(nfe), churn=int(churn), resid=int(resid),
                                opened=ws.get("opened"),
                                rewrites=ws.get("rewrites"),
                                gold=gold, out_prefix=text[:200]))
            if (i + 1) % 10 == 0:
                with LOCK:
                    RESULTS["gsm8k"][name] = dict(
                        mode=mode, window=window,
                        editing_threshold=editing_threshold, block_length=block_length,
                        threshold=threshold, max_post_steps=max_post_steps,
                        gen_length=gen_length, limit=n, partial=i + 1,
                        correct=correct, details=list(details))
                    dump()
        except torch.cuda.OutOfMemoryError:
            log(f"  {name}[{i}] OOM -> skipping")
            torch.cuda.empty_cache()
            details.append(dict(i=i, error="oom"))
        except Exception as e:  # noqa: BLE001
            log(f"  {name}[{i}] error: {e!r}")
            details.append(dict(i=i, error=str(e)))
    wall = time.time() - t0
    total = len([d for d in details if "error" not in d])
    if total > 0:
        with LOCK:
            EST[("gsm8k", mode, block_length)] = wall / total
    res = dict(
        mode=mode, window=window,
        editing_threshold=editing_threshold, block_length=block_length,
        threshold=threshold, max_post_steps=max_post_steps,
        gen_length=gen_length, limit=n, attempted=total,
        correct=correct,
        accuracy=100.0 * correct / max(total, 1),
        avg_tokens=tot_tok / max(total, 1),
        avg_nfe=tot_nfe / max(total, 1),
        avg_churn=tot_churn / max(total, 1),
        avg_resid=tot_resid / max(total, 1),
        wall_seconds=wall, details=list(details),
    )
    with LOCK:
        RESULTS["gsm8k"][name] = res
    log(f"== {name}: acc={res['accuracy']:.2f}% n={total} "
        f"avg_tok={res['avg_tokens']:.0f} avg_nfe={res['avg_nfe']:.1f} "
        f"avg_churn={res['avg_churn']:.1f} wall={wall:.0f}s")
    dump()


# ---------------------------------------------------------------- HumanEval
# HumanEval.jsonl embedded (gzip+base64), identical artifact to the v3 kernel
# — canonical OpenAI HumanEval, 164 problems. Embedded so the run does not
# depend on HF dataset availability or URI-format changes (push-3 failed on
# the openai_humaneval -> namespace/name migration).
HUMANEVAL_GZ_B64 = "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"


def load_humaneval():
    raw = gzip.decompress(base64.b64decode(HUMANEVAL_GZ_B64))
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def he_prompt_ids(doc, dev):
    """Saber (ACL 2026) prompt protocol, verbatim."""
    user = ("Write a solution to the following problem and make sure that it "
            "passes the tests:\n```python\n" + doc["prompt"] + "\n```")
    trailing = "Here is the completed function:\n```python\n" + doc["prompt"] + "\n"
    messages = [{"role": "user", "content": user}]
    p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + trailing
    return tok(p, return_tensors="pt").input_ids.to(dev)


def he_extract(gen_text, doc):
    """Saber's extract_answer, verbatim."""
    text_with_prefix = f"```python\n{doc['prompt']}\n" + gen_text
    entry_point = doc["entry_point"]
    pattern = re.compile(rf"def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL)
    match = pattern.search(text_with_prefix)
    if match:
        return match.group(1).rstrip()
    fallback = re.split(r"```", gen_text, maxsplit=1)
    if fallback:
        return fallback[0].rstrip()
    return gen_text


def _run_check_program(args):
    idx, program, timeout = args
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(program)
        proc = subprocess.Popen(
            [sys.executable, path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
            return idx, rc == 0, "ok"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except OSError:
                pass
            proc.wait()
            return idx, False, "timeout"
    except Exception as e:  # noqa: BLE001
        return idx, False, f"error: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def he_execute(details, docs, timeout=10, workers=8):
    jobs = []
    for d in details:
        if "error" in d or d.get("completion") is None:
            continue
        doc = docs[d["i"]]
        program = (doc["prompt"] + d["completion"] + "\n" + doc["test"]
                   + "\n" + f"check({doc['entry_point']})")
        jobs.append((d["i"], program, timeout))
    outcomes = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx, passed, info in ex.map(_run_check_program, jobs):
                outcomes[idx] = dict(passed=bool(passed), exec=info)
    for d in details:
        o = outcomes.get(d["i"])
        if o is None:
            d["passed"] = None
        else:
            d["passed"] = o["passed"]
            d["exec"] = o["exec"]
    return outcomes


def run_humaneval(name, model, mode="stock", window=16, editing_threshold=0.0,
                  block_length=32, threshold=0.5, max_post_steps=16,
                  gen_length=160, limit=164):
    docs = load_humaneval()
    n = min(limit, len(docs))
    log(f"humaneval {name}: n={n} block={block_length} thr={threshold} "
        f"edit_thr={editing_threshold} mps={max_post_steps}")
    t0 = time.time()
    tot_nfe = 0
    tot_tok = 0
    tot_churn = 0
    tot_resid = 0
    details = []
    for i, doc in enumerate(docs):
        if i >= n:
            break
        if remaining() < 900 and i > 0:
            log(f"  {name}: task-level early stop at {i}/{n} (time budget)")
            break
        ids = he_prompt_ids(doc, model.device)
        try:
            out, nfe, churn, resid = gen(
                model, ids, mode=mode, window=window, gen_length=gen_length,
                block_length=block_length, threshold=threshold,
                editing_threshold=editing_threshold,
                max_post_steps=max_post_steps,
            )
            gen_text = tok.decode(out[0], skip_special_tokens=True)
            completion = he_extract(gen_text, doc)
            tot_tok += int(out.numel())
            tot_nfe += nfe
            tot_churn += churn
            tot_resid += resid
            if i < 3 or (i + 1) % 20 == 0:
                log(f"  {name}[{i+1}/{n}] nfe={nfe} churn={churn} tok={int(out.numel())}")
            ws = model._nfe_state.get("wetink_stats") or {}
            details.append(dict(i=i, n_tokens=int(out.numel()), nfe=int(nfe),
                                churn=int(churn), resid=int(resid),
                                opened=ws.get("opened"),
                                rewrites=ws.get("rewrites"),
                                completion=completion[:600], full=gen_text[:300]))
            if (i + 1) % 10 == 0:
                with LOCK:
                    RESULTS["humaneval"][name] = dict(
                        mode=mode, window=window,
                        editing_threshold=editing_threshold, block_length=block_length,
                        threshold=threshold, max_post_steps=max_post_steps,
                        gen_length=gen_length, limit=n, partial=i + 1,
                        details=list(details))
                    dump()
        except torch.cuda.OutOfMemoryError:
            log(f"  {name}[{i}] OOM -> skipping")
            torch.cuda.empty_cache()
            details.append(dict(i=i, error="oom"))
        except Exception as e:  # noqa: BLE001
            log(f"  {name}[{i}] error: {e!r}")
            details.append(dict(i=i, error=str(e)))
    gen_wall = time.time() - t0
    total = len([d for d in details if "error" not in d])
    log(f"  {name}: gen done ({total} tasks, {gen_wall:.0f}s); executing checks ...")
    t1 = time.time()
    he_execute(details, docs, timeout=10)
    passed = sum(1 for d in details if d.get("passed"))
    exec_wall = time.time() - t1
    if total > 0:
        with LOCK:
            EST[("humaneval", mode, block_length)] = gen_wall / total
    res = dict(
        mode=mode, window=window,
        editing_threshold=editing_threshold, block_length=block_length,
        threshold=threshold, max_post_steps=max_post_steps,
        gen_length=gen_length, limit=n, attempted=total, passed=passed,
        pass_at_1=100.0 * passed / max(total, 1),
        avg_tokens=tot_tok / max(total, 1),
        avg_nfe=tot_nfe / max(total, 1),
        avg_churn=tot_churn / max(total, 1),
        avg_resid=tot_resid / max(total, 1),
        gen_wall_seconds=gen_wall, exec_wall_seconds=exec_wall,
        details=list(details),
    )
    with LOCK:
        RESULTS["humaneval"][name] = res
    log(f"== {name}: pass@1={res['pass_at_1']:.2f}% n={total} "
        f"avg_tok={res['avg_tokens']:.0f} avg_nfe={res['avg_nfe']:.1f} "
        f"avg_churn={res['avg_churn']:.1f} gen={gen_wall:.0f}s exec={exec_wall:.0f}s")
    dump()


# ---------------------------------------------------------------- run suite
# v5: two worker threads, one 4-bit model copy per GPU, pulling arms from a
# shared priority queue (same proven machinery as v4b). The arms are OUR
# wet-ink decoder at different window widths; baselines come from v4b.
#
#   P0 (core, must run):  wetink_w16            — the idea, T2T ON, W=16
#                          wetink_noedit_w16     — frontier-only control (T2T
#                                                 OFF): isolates the effect of
#                                                 the one-token frontier itself
#   P1 (window ablation): wetink_w8 (narrower), wetink_w32 (whole block wet;
#                          bridges to stock T2T scope)
#   P2 (transfer):        he_wetink_w16 (HumanEval, small n)
ARMS = [
    # (name, bench, mode, window, editing_threshold, block, threshold, mps, gen_len, cap)
    ("wetink_w16",        "gsm8k",     "wetink", 16, 0.0, 32, 0.5, 4, 160, 45),
    ("wetink_noedit_w16", "gsm8k",     "wetink", 16, 2.0, 32, 0.5, 4, 160, 45),
    ("wetink_w8",         "gsm8k",     "wetink",  8, 0.0, 32, 0.5, 4, 160, 30),
    ("wetink_w32",        "gsm8k",     "wetink", 32, 0.0, 32, 0.5, 4, 160, 30),
    ("he_wetink_w16",     "humaneval", "wetink", 16, 0.0, 32, 0.5, 4, 160, 25),
]
_QUEUE = Queue()
for _a in ARMS:
    _QUEUE.put(_a)

RUN_FN = {"gsm8k": run_gsm8k, "humaneval": run_humaneval}


def _worker_loop(wid, model):
    with ACT_LOCK:
        ACTIVE["n"] += 1
    try:
        while remaining() > 900:
            try:
                arm = _QUEUE.get_nowait()
            except Empty:
                break
            (name, bench, mode, window, et, bl, thr, mps, gl, cap) = arm
            prev = RESULTS[bench].get(name)
            if prev and (prev.get("attempted") or 0) > 0 and "error" not in prev:
                continue  # already done — do not rerun
            n = plan_n(bench, mode, bl, cap)
            if n < 8:
                with LOCK:
                    RESULTS[bench][name] = {"skipped": "time budget"}
                    dump()
                continue
            log(f"[w{wid}] arm {name}: n={n} mode={mode} W={window} block={bl} "
                f"edit_thr={et} thr={thr} mps={mps}")
            try:
                RUN_FN[bench](name, model, mode=mode, window=window,
                              editing_threshold=et, block_length=bl,
                              threshold=thr, max_post_steps=mps,
                              gen_length=gl, limit=n)
            except Exception as e:  # noqa: BLE001
                log(f"[w{wid}] arm {name} CRASHED: {e!r}")
                with LOCK:
                    RESULTS[bench][name] = {"error": str(e)}
                    dump()
    finally:
        with ACT_LOCK:
            ACTIVE["n"] -= 1
        log(f"[w{wid}] worker finished (remaining={remaining():.0f}s)")


def _worker_main(wid):
    if wid == 0:
        _worker_loop(0, M0)
        return
    log("[w1] loading model #1 in 4-bit NF4 on GPU1 ...")
    t0 = time.time()
    try:
        m1 = _load(1)
        _attach_meter(m1)
        log(f"[w1] model #1 loaded in {time.time()-t0:.0f}s")
    except Exception as e:  # noqa: BLE001
        log(f"[w1] model #1 load FAILED: {e!r} -> single-worker mode")
        return
    _worker_loop(1, m1)


_THREADS = [threading.Thread(target=_worker_main, args=(i,)) for i in range(2)]
for _t in _THREADS:
    _t.start()
for _t in _THREADS:
    _t.join()


# ---------------------------------------------------------------- paired stats
def _pair(stock_arm, wet_arm):
    det_s = [d for d in stock_arm.get("details", []) if "error" not in d]
    det_w = [d for d in wet_arm.get("details", []) if "error" not in d]
    s_map = {d["i"]: d for d in det_s}
    w_map = {d["i"]: d for d in det_w}
    common = sorted(set(s_map) & set(w_map))
    b = c = 0
    for i in common:
        os_, ow_ = ok_of(s_map[i]), ok_of(w_map[i])
        if not os_ and ow_:
            b += 1
        elif os_ and not ow_:
            c += 1
    k = sum(1 for d in det_w if ok_of(d))
    n = len(det_w)
    p, lo, hi = wilson(k, n)
    sk = sum(1 for d in det_s if ok_of(d))
    sn = len(det_s)
    return dict(
        n=n, k=k, acc=100.0 * k / max(n, 1),
        ci95=[100.0 * lo, 100.0 * hi],
        fixed=b, broken=c, mcnemar_p=mcnemar_exact(b, c),
        avg_nfe=sum(d.get("nfe", 0) for d in det_w) / max(n, 1),
        stock_acc=100.0 * sk / max(sn, 1),
        stock_avg_nfe=sum(d.get("nfe", 0) for d in det_s) / max(sn, 1),
        nfe_ratio=(sum(d.get("nfe", 0) for d in det_w) / max(n, 1)) /
                  (sum(d.get("nfe", 0) for d in det_s) / max(sn, 1))
        if sn and sum(d.get("nfe", 0) for d in det_s) else float("nan"),
    )


def paired_stats(bench):
    """In-run pairing: every T2T-ON wet-ink arm vs the frontier-only control
    (wetink_noedit_w16, editing_threshold>1). Cross-run pairing vs the v4b
    baselines (stock_b32 / noedit_b32) is done offline by analyze_v5.py."""
    arms = RESULTS[bench]
    ctrl = None
    for name, r in arms.items():
        if "skipped" in r or "error" in r or "details" not in r:
            continue
        if r.get("editing_threshold", 0.0) > 1.0:
            ctrl = (name, r)
            break
    out = {}
    if ctrl is not None:
        for name, r in arms.items():
            if r.get("editing_threshold", 0.0) > 1.0:
                continue
            if "skipped" in r or "error" in r or "details" not in r:
                continue
            entry = _pair(ctrl[1], r)
            entry["vs"] = ctrl[0]
            out[name] = entry
    RESULTS["stats"][bench] = {
        "control": ctrl[0] if ctrl else None,
        "vs_control": out,
    }


paired_stats("gsm8k")
paired_stats("humaneval")

# ---------------------------------------------------------------- summary
log("=========== SUMMARY v5 (wet-ink: OUR decoder) ===========")
for bench, metric in (("gsm8k", "accuracy"), ("humaneval", "pass_at_1")):
    log(f"--- {bench} ---")
    for name, r in RESULTS[bench].items():
        if "skipped" in r:
            log(f"{name:<24} SKIPPED")
            continue
        if "error" in r:
            log(f"{name:<24} ERROR: {str(r['error'])[:90]}")
            continue
        wall = r.get("wall_seconds", r.get("gen_wall_seconds", 0))
        log(f"{name:<24} {metric}={r[metric]:6.2f}%  n={r['attempted']:>3} "
            f"nfe={r['avg_nfe']:7.1f} churn={r.get('avg_churn',0):6.1f} "
            f"tok={r['avg_tokens']:5.0f} wall={wall:6.0f}s "
            f"W={r.get('window')} et={r.get('editing_threshold')}")
    st = RESULTS["stats"].get(bench, {})
    for name, s in st.get("vs_control", {}).items():
        log(f"  {name:<22} vs {s['vs']:<18} acc={s['acc']:6.2f}% "
            f"CI[{s['ci95'][0]:.1f},{s['ci95'][1]:.1f}] fixed/broken={s['fixed']}/"
            f"{s['broken']} McNemar p={s['mcnemar_p']:.4f}"
            + (f" NFE x{s['nfe_ratio']:.2f}" if s.get("nfe_ratio") else ""))
dump()
log("DONE. results_v5.json written to", RESULTS_PATH)
