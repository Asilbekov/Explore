#!/usr/bin/env python3
"""Analyze wet-ink v4 benchmark results (LLaDA2.2-mini, adaptive-N suite).

Reads results/results_v4.json produced by the Kaggle kernel, recomputes
per-arm accuracy with Wilson CIs, paired McNemar exact tests vs the matching
noedit arm (same benchmark + block size), NFE/churn accounting on the PAIRED
subset, truncation diagnostics (gen_length=160 cap), and budget accounting.
Cross-checks the in-kernel stats section.

Usage: python3 scripts/analyze_v4.py [path/to/results_v4.json]
"""
import json
import math
import sys
from math import comb

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/results_v4.json"
TOK_CAP = 160  # gen_length of the v4 GSM8K arms


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact binomial (McNemar) test on discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def ok_of(d):
    v = d.get("ok", d.get("passed"))
    return bool(v)


def boxed_in(text):
    return "\\boxed" in text


def main():
    with open(PATH) as f:
        R = json.load(f)

    meta = R.get("meta", {})
    print("=" * 100)
    print("WET-INK v4 RESULTS  |  model:", meta.get("model"),
          "| gpu:", meta.get("gpu"), "| suite:", meta.get("suite"))
    print("=" * 100)

    # ---- sanity + equivalence ----
    san = R.get("sanity", {})
    if san:
        print("\n--- sanity ---")
        for k, v in san.items():
            print(f"  {k}: nfe={v.get('nfe')} churn={v.get('churn')} "
                  f"special_residual={v.get('special_residual')} "
                  f"wall={v.get('wall_seconds', 0):.0f}s")
            print(f"    out[:120]: {v.get('output_prefix', '')[:120]!r}")

    eq = R.get("equivalence", {})
    if eq:
        print("\n--- equivalence (greedy determinism, must all be ids_equal=True) ---")
        for k, v in eq.items():
            print(f"  {k}: ids_equal={v.get('ids_equal')} nfe {v.get('nfe1')}=={v.get('nfe2')} "
                  f"churn {v.get('churn1')}=={v.get('churn2')}")

    all_rows = {}
    for bench in ("gsm8k", "humaneval"):
        arms = R.get(bench, {})
        if not arms:
            continue
        print(f"\n{'=' * 100}\n{bench.upper()}\n{'=' * 100}")
        rows = {}
        for name, a in arms.items():
            if "skipped" in a:
                print(f"  {name:26s} SKIPPED (time budget)")
                continue
            det = [d for d in a.get("details", []) if "error" not in d]
            n = len(det)
            if n == 0:
                print(f"  {name}: NO VALID DETAILS")
                continue
            k = sum(ok_of(d) for d in det)
            p, lo, hi = wilson(k, n)
            nfe = sum(d.get("nfe", 0) for d in det) / n
            churn = sum(d.get("churn", 0) for d in det) / n
            tok = sum(d.get("n_tokens", 0) for d in det) / n
            blk = a.get("block_length")
            wall = a.get("wall_seconds",
                         a.get("gen_wall_seconds", 0) + a.get("exec_wall_seconds", 0))
            trunc = sum(1 for d in det if d.get("n_tokens", 0) >= TOK_CAP) \
                if bench == "gsm8k" else None
            bx = sum(1 for d in det if boxed_in(d.get("out_prefix", ""))) \
                if bench == "gsm8k" else None
            rows[name] = dict(det=det, n=n, k=k, p=p, lo=lo, hi=hi, nfe=nfe,
                              churn=churn, tok=tok, blk=blk, wall=wall,
                              trunc=trunc, boxed=bx, cfg=a)
            all_rows[(bench, name)] = rows[name]
            t = f" trunc@cap={trunc}({100*trunc/n:.0f}%) boxed={bx}({100*bx/n:.0f}%)" \
                if trunc is not None else ""
            print(f"  {name:26s} n={n:3d}  acc={100*p:5.1f}%  "
                  f"CI[{100*lo:4.1f},{100*hi:4.1f}]  nfe={nfe:6.1f}  "
                  f"churn={churn:5.2f}  tok={tok:5.1f}  "
                  f"wall={wall:6.0f}s ({wall/n:5.0f}s/task)  block={blk}{t}")

        # paired tests vs noedit with same block size
        print(f"\n  --- paired vs noedit (same block size) ---")
        for name, r in rows.items():
            if name.startswith("noedit") or "_noedit" in name:
                continue
            blk = r["blk"]
            base_name = next((s for s in (f"noedit_b{blk}", f"he_noedit_b{blk}")
                              if s in rows), None)
            if base_name is None:
                cand = [s for s in rows
                        if s.startswith(("noedit", "he_noedit"))
                        and rows[s]["blk"] == blk]
                base_name = cand[0] if cand else None
            if base_name is None:
                continue
            base = {d["i"]: d for d in rows[base_name]["det"]}
            cur = {d["i"]: d for d in r["det"]}
            common = sorted(set(base) & set(cur))
            b = c = 0  # b: noedit wrong & arm right; c: noedit right & arm wrong
            for i in common:
                os_, oc_ = ok_of(base[i]), ok_of(cur[i])
                if not os_ and oc_:
                    b += 1
                elif os_ and not oc_:
                    c += 1
            pv = mcnemar_exact(b, c)
            # paired NFE/churn on common items (fair comparison)
            nfe_c = sum(cur[i].get("nfe", 0) for i in common) / max(len(common), 1)
            nfe_b = sum(base[i].get("nfe", 0) for i in common) / max(len(common), 1)
            churn_c = sum(cur[i].get("churn", 0) for i in common) / max(len(common), 1)
            churn_b = sum(base[i].get("churn", 0) for i in common) / max(len(common), 1)
            nfe_ratio = nfe_c / nfe_b if nfe_b else float("nan")
            dp = (100 * r["k"] / r["n"]
                  - 100 * rows[base_name]["k"] / rows[base_name]["n"])
            star = " *" if pv < 0.05 else ""
            print(f"  {name:26s} vs {base_name:16s} (n={len(common):3d}): "
                  f"diff={dp:+5.1f}pp  win/loss={b}/{c}  McNemar p={pv:.4f}{star}  "
                  f"NFE x{nfe_ratio:.2f} (paired {nfe_b:.1f}->{nfe_c:.1f})  "
                  f"churn {churn_b:.2f}->{churn_c:.2f}")

    # ---- truncation effect: accuracy among finished vs truncated ----
    g4 = R.get("gsm8k", {})
    if g4:
        print(f"\n{'=' * 100}\nTRUNCATION DIAGNOSTICS (gen_length={TOK_CAP})\n{'=' * 100}")
        for name, a in g4.items():
            det = [d for d in a.get("details", []) if "error" not in d]
            if not det:
                continue
            fin = [d for d in det if d.get("n_tokens", 0) < TOK_CAP]
            tru = [d for d in det if d.get("n_tokens", 0) >= TOK_CAP]
            kf, nf = sum(ok_of(d) for d in fin), len(fin)
            kt, nt = sum(ok_of(d) for d in tru), len(tru)
            pf = wilson(kf, nf) if nf else (0, 0, 0)
            pt = wilson(kt, nt) if nt else (0, 0, 0)
            print(f"  {name:26s} finished(EOS): n={nf:3d} acc={100*pf[0]:5.1f}% "
                  f"CI[{100*pf[1]:4.1f},{100*pf[2]:4.1f}]   "
                  f"truncated@{TOK_CAP}: n={nt:3d} acc={100*pt[0]:5.1f}% "
                  f"CI[{100*pt[1]:4.1f},{100*pt[2]:4.1f}]")

    # ---- budget accounting ----
    print(f"\n{'=' * 100}\nBUDGET ACCOUNTING\n{'=' * 100}")
    tot = 0.0
    for (bench, name), r in sorted(all_rows.items()):
        h = r["wall"] / 3600
        tot += h
        print(f"  {bench}/{name:22s} wall={h:5.2f}h  n={r['n']:3d}  "
              f"{r['wall']/r['n']:5.0f}s/task")
    san_w = sum(v.get("wall_seconds", 0) for v in R.get("sanity", {}).values()) / 3600
    print(f"  sanity+equivalence+load ~{san_w:.2f}h + fixed overhead")
    print(f"  sum of measured arm walls: {tot:.2f}h (kernel session budget 7.5h)")

    # ---- in-kernel stats cross-check ----
    st = R.get("stats", {})
    if st:
        print(f"\n{'=' * 100}\nIN-KERNEL STATS SECTION (cross-check)\n{'=' * 100}")
        print(json.dumps(st, indent=1)[:3000])

    print("\n(*) p < 0.05, exact McNemar on paired task outcomes.")


if __name__ == "__main__":
    main()
