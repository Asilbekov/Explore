"""CPU smoke test of wet_ink v2 on a tiny random-weight Nemotron-Labs-Diffusion
(the EXACT architecture files of the 3B model).

Checks:
  1.  stock model.generate() runs on the tiny model;
  2.  EQUIVALENCE: generate_wet_ink(revision_blocks=0) is bit-identical to stock
      generate() (ids AND nfe), including the EOS early-stop path — for every
      policy (revision_blocks=0 forces the stock path);
  3.  reme_di (RemeDi loop): runs, no [MASK] left, EXACTLY denoise_steps
      forwards per block (termination by construction), revision happens;
  4.  reme_di with criterion='entropy' (Dream/NVlabs score) runs and ranks
      differently from 'confidence';
  5.  codr: runs, no [MASK] left, probe forwards == num_partitions per
      windowed block, drift remasking fires on a random model, bounded NFE;
  6.  per_step (v1 legacy): still runs (churn + tail expected on random
      weights) — kept for reproducibility of the v1 measurements;
  7.  KV-CACHE INVARIANT for the new policies: after crop+rebuild the final
      cache equals a fresh causal forward over the final sequence;
  8.  batch B=2 equivalence; determinism at temperature 0.
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

    # ---- 2. equivalence W=0 (all policies force the stock path) ----
    for pol in ("none", "reme_di", "codr", "per_step"):
        out0, nfe0 = generate_wet_ink(
            model, prompt, revision_blocks=0, remask_policy=pol, **gen_kwargs
        )
        check(f"W=0 policy={pol}: identical ids to stock",
              bool(torch.equal(out_stock, out0)))
        check(f"W=0 policy={pol}: identical nfe to stock",
              nfe_stock == nfe0, f"{nfe_stock} vs {nfe0}")

    # EOS early-stop path
    eos_probe = int(out_stock[0, 20].item())
    out_stock_eos, nfe_stock_eos = model.generate(
        prompt, eos_token_id=eos_probe, **gen_kwargs)
    out0_eos, nfe0_eos = generate_wet_ink(
        model, prompt, revision_blocks=0, eos_token_id=eos_probe, **gen_kwargs
    )
    check("W=0 EOS path identical ids", bool(torch.equal(out_stock_eos, out0_eos)))
    check("W=0 EOS path identical nfe", nfe_stock_eos == nfe0_eos,
          f"{nfe_stock_eos} vs {nfe0_eos}")
    check("EOS truncation actually fired", out_stock_eos.shape[1] < 16 + 64,
          f"len={out_stock_eos.shape[1]}")

    # ---- 3. reme_di ----
    out_r, nfe_r, st_r = generate_wet_ink(
        model, prompt, revision_blocks=1, remask_policy="reme_di",
        return_stats=True, **gen_kwargs
    )
    check("reme_di W=1 runs, no [MASK] left", not bool((out_r == mask_id).any()))
    check("reme_di remasking fired (re-liquefaction)", st_r["remask_events"] > 0,
          f"remask_events={st_r['remask_events']}")
    check("reme_di revised tokens in window", st_r["revision_token_changes"] > 0,
          f"changes={st_r['revision_token_changes']}")
    # termination by construction: exactly denoise_steps denoise forwards per
    # block (2 blocks here: block 0 has no window, block 1 has W=1) + 1
    # post-block forward each
    n_blocks = 64 // 32
    check("reme_di exact step count (no tail, no cap)",
          st_r["denoise_forwards"] == n_blocks * 32,
          f"denoise_fw={st_r['denoise_forwards']} expected={n_blocks * 32}")
    check("reme_di no safety tail", st_r["safety_tail_steps"] == 0)
    check("reme_di no hard caps", st_r["hard_cap_hits"] == 0)
    check("reme_di NFE bound", nfe_r <= n_blocks * (32 + 1),
          f"nfe={nfe_r} <= {n_blocks * 33}")
    print(f"       reme_di stats: remask={st_r['remask_events']}, "
          f"changes={st_r['revision_token_changes']}, "
          f"denoise_fw={st_r['denoise_forwards']}, post_fw={st_r['post_block_forwards']}, "
          f"nfe={nfe_r} (stock {nfe_stock})")

    # ---- 4. reme_di + entropy criterion ----
    out_e, nfe_e, st_e = generate_wet_ink(
        model, prompt, revision_blocks=1, remask_policy="reme_di",
        criterion="entropy", return_stats=True, **gen_kwargs
    )
    check("reme_di entropy runs, no [MASK]", not bool((out_e == mask_id).any()))
    check("reme_di entropy exact step count",
          st_e["denoise_forwards"] == n_blocks * 32,
          f"denoise_fw={st_e['denoise_forwards']}")
    check("reme_di entropy ranks differently (random model)",
          not bool(torch.equal(out_r, out_e)),
          "outputs differ" if not torch.equal(out_r, out_e) else "SAME (suspicious)")

    # ---- 5. codr ----
    out_c, nfe_c, st_c = generate_wet_ink(
        model, prompt, revision_blocks=1, remask_policy="codr",
        return_stats=True, **gen_kwargs
    )
    check("codr W=1 runs, no [MASK] left", not bool((out_c == mask_id).any()))
    check("codr probes == partitions per windowed block",
          st_c["codr_probe_forwards"] == 1 * 2,  # only block 1 has a window
          f"probes={st_c['codr_probe_forwards']}")
    check("codr drift remasking fired", st_c["codr_drift_remasks"] > 0,
          f"drift_remasks={st_c['codr_drift_remasks']}")
    check("codr no safety tail", st_c["safety_tail_steps"] == 0)
    check("codr bounded NFE",
          nfe_c <= n_blocks * (32 + 1) + st_c["codr_probe_forwards"] + 64,
          f"nfe={nfe_c}")
    print(f"       codr stats: probes={st_c['codr_probe_forwards']}, "
          f"drift={st_c['codr_drift_remasks']}, refill_fw={st_c['codr_refill_forwards']}, "
          f"nfe={nfe_c}")

    # ---- 6. per_step (v1 legacy) ----
    out_p, nfe_p, st_p = generate_wet_ink(
        model, prompt, revision_blocks=1, remask_policy="per_step",
        return_stats=True, **gen_kwargs
    )
    check("per_step W=1 runs, no [MASK] left", not bool((out_p == mask_id).any()))
    check("per_step remasking fired", st_p["remask_events"] > 0)
    check("per_step v1 churn present on random model (documented)",
          st_p["safety_tail_steps"] > 0 or st_p["hard_cap_hits"] > 0,
          f"tail={st_p['safety_tail_steps']}, caps={st_p['hard_cap_hits']}")
    print(f"       per_step stats: remask={st_p['remask_events']}, "
          f"tail={st_p['safety_tail_steps']}, caps={st_p['hard_cap_hits']}, "
          f"nfe={nfe_p} (=> the v1 behaviour, for reproducibility)")

    # ---- 7. KV-cache invariant (crop + rebuild) for both new policies ----
    for pol in ("reme_di", "codr"):
        out_x, _, st_x = generate_wet_ink(
            model, prompt, revision_blocks=1, remask_policy=pol,
            return_stats=True, **gen_kwargs
        )
        gen_cache = st_x["_final_cache"]
        for layer in model.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = False
        fresh = model(out_x, use_cache=True, use_causal_mask=True)
        fresh_cache = fresh.past_key_values
        ok = True
        max_err = 0.0
        for l in range(len(gen_cache.layers)):
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
        check(f"cache invariant ({pol}): final cache == fresh causal forward",
              ok, f"max_abs_err={max_err:.2e}")

    # ---- 8. batch=2 + determinism ----
    prompt2 = torch.randint(200, 2000, (2, 12))
    outs, nfes = model.generate(prompt2, **gen_kwargs)
    outb, nfeb = generate_wet_ink(
        model, prompt2, revision_blocks=0, remask_policy="reme_di", **gen_kwargs
    )
    check("B=2 W=0 identical ids", bool(torch.equal(outs, outb)))
    check("B=2 W=0 identical nfe", nfes == nfeb)

    outb1, _, stb1 = generate_wet_ink(
        model, prompt2, revision_blocks=1, remask_policy="reme_di",
        return_stats=True, **gen_kwargs
    )
    check("B=2 reme_di runs, no [MASK]", not bool((outb1 == mask_id).any()),
          f"remask={stb1['remask_events']}")

    out_r2, _, _ = generate_wet_ink(
        model, prompt, revision_blocks=1, remask_policy="reme_di",
        return_stats=True, **gen_kwargs
    )
    check("reme_di deterministic at temp=0", bool(torch.equal(out_r, out_r2)))

    print()
    if FAILURES:
        print("FAILED:", FAILURES)
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
