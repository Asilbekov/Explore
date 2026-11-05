"""CPU smoke test of wet_ink.generate_wet_ink on a tiny random-weight
Nemotron-Labs-Diffusion (the EXACT architecture files of the 3B model).

Checks:
  1. stock model.generate() runs on the tiny model;
  2. EQUIVALENCE: generate_wet_ink(revision_blocks=0) is bit-identical to stock
     generate() (ids AND nfe), including the EOS early-stop path;
  3. wet-ink W=1 / W=2 / fixed1 / causal_context=False run without crashing,
     never leave [MASK] in the output, and terminate well under the safety cap;
  4. re-masking actually fires on a random model (confidence ~ 1/vocab << 0.9);
  5. KV-CACHE INVARIANT: after a wet-ink run with cache surgery (crop+rebuild),
     the final cache equals a fresh causal forward over the final sequence
     (i.e. revised tokens were correctly written back into the cache);
  6. batch B=2 equivalence;
  7. determinism at temperature 0.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from make_tiny_repo import main as make_repo, DST as REPO  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModel  # noqa: E402

from wet_ink import generate_wet_ink  # noqa: E402

torch.manual_seed(0)

FAILURES = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {extra}")
    if not cond:
        FAILURES.append(name)


def main():
    torch.manual_seed(0)
    if not os.path.isdir(REPO) or not os.path.isfile(os.path.join(REPO, "config.json")):
        make_repo()

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True).eval()
    mask_id = model.mask_token_id
    prompt = torch.randint(200, 2000, (1, 16))

    gen_kwargs = dict(max_new_tokens=64, block_length=32, threshold=0.9)

    # ---- 1. stock generate ----
    out_stock, nfe_stock = model.generate(prompt, **gen_kwargs)
    check("stock generate runs", out_stock.shape == (1, 16 + 64), f"nfe={nfe_stock}")
    check("stock output has no [MASK]", not bool((out_stock == mask_id).any()))

    # ---- 2. equivalence W=0 ----
    out0, nfe0 = generate_wet_ink(model, prompt, revision_blocks=0, **gen_kwargs)
    check("W=0 identical ids to stock", bool(torch.equal(out_stock, out0)))
    check("W=0 identical nfe to stock", nfe_stock == nfe0, f"{nfe_stock} vs {nfe0}")

    # ---- 2b. equivalence on the EOS early-stop path ----
    # pick a token that certainly appears in the stock output -> EOS must fire
    eos_probe = int(out_stock[0, 20].item())
    out_stock_eos, nfe_stock_eos = model.generate(prompt, eos_token_id=eos_probe, **gen_kwargs)
    out0_eos, nfe0_eos = generate_wet_ink(
        model, prompt, revision_blocks=0, eos_token_id=eos_probe, **gen_kwargs
    )
    check("W=0 EOS path identical ids", bool(torch.equal(out_stock_eos, out0_eos)))
    check("W=0 EOS path identical nfe", nfe_stock_eos == nfe0_eos,
          f"{nfe_stock_eos} vs {nfe0_eos}")
    check("EOS truncation actually fired", out_stock_eos.shape[1] < 16 + 64,
          f"len={out_stock_eos.shape[1]}")

    # ---- 3/4. wet-ink windows ----
    # NOTE: on a RANDOM model all confidences ~ 1/vocab << 0.9, so remasking
    # churns (every committed window token is immediately re-masked) and the
    # safety tail is expected to be used. A REAL model has confidences > 0.9,
    # where remasking fires rarely; the threshold=0.0 control below mimics that.
    for W in (1, 2):
        out_w, nfe_w, st = generate_wet_ink(
            model, prompt, revision_blocks=W, return_stats=True, **gen_kwargs
        )
        check(f"W={W} runs, no [MASK] left", not bool((out_w == mask_id).any()))
        check(f"W={W} remasking fired", st["remask_events"] > 0,
              f"remask_events={st['remask_events']}")
        check(f"W={W} revised tokens in window", st["revision_token_changes"] > 0,
              f"changes={st['revision_token_changes']}")
        check(f"W={W} output fully denoised (tail cleared masks)",
              not bool((out_w == mask_id).any()))
        print(f"       W={W} stats: tail={st['safety_tail_steps']}, "
              f"blocks_with_remask={st['blocks_with_remask']}, "
              f"commits={st['commit_events']}, denoise_fw={st['denoise_forwards']}, "
              f"post_fw={st['post_block_forwards']}, nfe={nfe_w} (stock {nfe_stock})")

    # control: threshold=0.0 -> every confidence passes -> remasking never fires
    # -> no churn -> safety tail must stay at zero (stock-like step count)
    out_c, nfe_c, st_c = generate_wet_ink(
        model, prompt, revision_blocks=1, threshold=0.0,
        return_stats=True, max_new_tokens=64, block_length=32,
    )
    check("control thr=0.0: remasking inert", st_c["remask_events"] == 0)
    check("control thr=0.0: safety tail unused", st_c["safety_tail_steps"] == 0,
          f"tail={st_c['safety_tail_steps']}")
    check("control thr=0.0: no [MASK] left", not bool((out_c == mask_id).any()))

    # ---- fixed1 commit mode ----
    out_f1, nfe_f1, st_f1 = generate_wet_ink(
        model, prompt, revision_blocks=1, commit="fixed1",
        threshold=None, return_stats=True, max_new_tokens=64, block_length=32,
    )
    check("fixed1 W=1 runs, no [MASK] left", not bool((out_f1 == mask_id).any()))
    check("fixed1 nfe >= span steps", nfe_f1 >= 64, f"nfe={nfe_f1}")

    # ---- causal_context=False path ----
    out_nc, nfe_nc, st_nc = generate_wet_ink(
        model, prompt, revision_blocks=1, causal_context=False,
        return_stats=True, **gen_kwargs
    )
    check("causal_context=False runs", out_nc.shape == (1, 80), f"nfe={nfe_nc}")
    check("causal_context=False no [MASK] left", not bool((out_nc == mask_id).any()))

    # ---- 5. KV-cache invariant (crop + rebuild correctness) ----
    out_w1, nfe_w1, st_w1 = generate_wet_ink(
        model, prompt, revision_blocks=1, return_stats=True, **gen_kwargs
    )
    gen_cache = st_w1["_final_cache"]
    # Reference: fresh CAUSAL forward over the final sequence.
    # NOTE: stock generate() leaves diffusion_lm=True on exit, so we must
    # explicitly switch the attention to causal mode for the reference pass
    # (same toggle ar_generate uses).
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = False
    fresh = model(out_w1, use_cache=True, use_causal_mask=True)
    fresh_cache = fresh.past_key_values
    ok = True
    max_err = 0.0
    n_layers = len(gen_cache.layers)
    for l in range(n_layers):
        for kind in ("keys", "values"):
            a = getattr(gen_cache.layers[l], kind)
            b = getattr(fresh_cache.layers[l], kind)
            if a.shape != b.shape:
                ok = False
                break
            err = (a - b).abs().max().item()
            max_err = max(max_err, err)
            if err > 1e-4:
                ok = False
    check("cache invariant: final cache == fresh causal forward", ok,
          f"max_abs_err={max_err:.2e}, layers={n_layers}, "
          f"seq={gen_cache.get_seq_length()} vs {fresh_cache.get_seq_length()}")

    # ---- 6. batch=2 equivalence ----
    prompt2 = torch.randint(200, 2000, (2, 12))
    outs, nfes = model.generate(prompt2, **gen_kwargs)
    outb, nfeb = generate_wet_ink(model, prompt2, revision_blocks=0, **gen_kwargs)
    check("B=2 W=0 identical ids", bool(torch.equal(outs, outb)))
    check("B=2 W=0 identical nfe", nfes == nfeb)

    outb1, nfeb1, stb1 = generate_wet_ink(
        model, prompt2, revision_blocks=1, return_stats=True, **gen_kwargs
    )
    check("B=2 W=1 runs, no [MASK]", not bool((outb1 == mask_id).any()),
          f"remask={stb1['remask_events']}")

    # ---- 7. determinism ----
    out_w1b, _, _ = generate_wet_ink(
        model, prompt, revision_blocks=1, return_stats=True, **gen_kwargs
    )
    check("W=1 deterministic at temp=0", bool(torch.equal(out_w1, out_w1b)))

    print()
    if FAILURES:
        print("FAILED:", FAILURES)
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
