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
