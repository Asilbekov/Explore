#!/usr/bin/env python3
"""Analyze wet-ink v4 results (LLaDA2.2-mini, trained T2T gate).

Reads results_v4.json produced by the kaggle_v4 kernel. The CONTROL is the
noedit arm (editing_threshold=2.0 -> T2T OFF); treatments are stock/edit arms
(T2T ON at various gate levels). Computes Wilson CIs, paired exact McNemar vs
the noedit control at the same block length, NFE and churn accounting.

Usage: python3 scripts/analyze_v4.py [path/to/results_v4.json]
"""
import json
import math
import sys
from math import comb

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/wet_ink/results/results_v4.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def ok_of(d):
    return bool(d.get("ok", d.get("passed")))


def metric_of(arm):
    return "accuracy" if "accuracy" in arm else "pass_at_1"


def rows_of(R, bench):
    rows = {}
    for name, a in R.get(bench, {}).items():
        if "skipped" in a or "details" not in a:
            continue
        det = [d for d in a["details"] if "error" not in d]
        n = len(det)
        if n == 0:
            continue
        k = sum(ok_of(d) for d in det)
        p, lo, hi = wilson(k, n)
        rows[name] = dict(
            det=det, n=n, k=k, p=p, lo=lo, hi=hi,
            nfe=sum(d.get("nfe", 0) for d in det) / n,
            churn=sum(d.get("churn", 0) for d in det) / n,
            tok=sum(d.get("n_tokens", 0) for d in det) / n,
            blk=a.get("block_length"), cfg=a,
        )
    return rows


def main():
    with open(PATH) as f:
        R = json.load(f)

    meta = R.get("meta", {})
    print("=" * 104)
    print("WET-INK v4 RESULTS | model:", meta.get("model"),
          "| quant:", meta.get("quantization"),
          "| attn:", meta.get("attn_impl"))
    print("suite:", meta.get("suite"))
    print("=" * 104)

    eq = R.get("equivalence", {})
    if eq:
        print("\n--- greedy determinism (same config twice -> identical ids) ---")
        for k, v in eq.items():
            print(f"  {k}: {v}")

    san = R.get("sanity", {})
    if san:
        print("\n--- sanity ---")
        for k, v in san.items():
            print(f"  {k}: nfe={v.get('nfe')} churn={v.get('churn')} "
                  f"resid={v.get('special_residual')} wall={v.get('wall_seconds', 0):.0f}s")
            print(f"    out: {v.get('output_prefix', '')[:140]!r}")

    for bench in ("gsm8k", "humaneval"):
        rows = rows_of(R, bench)
        if not rows:
            continue
        print(f"\n{'=' * 104}\n{bench.upper()}\n{'=' * 104}")
        m = "acc" if bench == "gsm8k" else "pass@1"
        for name, r in sorted(rows.items()):
            print(f"  {name:22s} n={r['n']:3d}  {m}={100*r['p']:5.1f}%  "
                  f"CI[{100*r['lo']:4.1f},{100*r['hi']:4.1f}]  "
                  f"nfe={r['nfe']:6.1f} churn={r['churn']:6.1f} "
                  f"tok={r['tok']:5.0f} blk={r['blk']}")

        # controls = noedit arms; everything else pairs vs noedit at same block
        controls = {n: r for n, r in rows.items()
                    if n.startswith("noedit") or n.startswith("he_noedit")}
        print(f"\n  --- paired vs noedit control (same block): does the trained gate help? ---")
        for name, r in sorted(rows.items()):
            if name in controls:
                continue
            blk = r["blk"]
            cname = next((c for c in controls if controls[c]["blk"] == blk), None)
            if cname is None:
                continue
            ctl = controls[cname]
            cmap = {d["i"]: d for d in ctl["det"]}
            tmap = {d["i"]: d for d in r["det"]}
            common = sorted(set(cmap) & set(tmap))
            b = c = 0  # b: control wrong & treatment right; c: converse
            for i in common:
                oc, ot = ok_of(cmap[i]), ok_of(tmap[i])
                if not oc and ot:
                    b += 1
                elif oc and not ot:
                    c += 1
            pv = mcnemar_exact(b, c)
            dp = 100 * (r["k"] / r["n"] - ctl["k"] / ctl["n"])
            star = " *" if pv < 0.05 else ""
            nfe_ratio = (r["nfe"] / ctl["nfe"]) if ctl["nfe"] else None
            ch_ratio = (r["churn"] / ctl["churn"]) if ctl["churn"] else None
            nfe_r = f"x{nfe_ratio:.2f}" if nfe_ratio is not None else "—"
            ch_r = f"x{ch_ratio:.2f}" if ch_ratio is not None else "n/a(0)"
            print(f"  {name:22s} vs {cname:14s} (n={len(common):3d}): "
                  f"diff={dp:+5.1f}pp  fix/brk={b}/{c}  McNemar p={pv:.4f}{star}  "
                  f"NFE {nfe_r} churn {ch_r}")

        # block effect on the control itself
        ref = controls.get("noedit_b32") or controls.get("he_noedit_b32")
        if ref:
            print(f"\n  --- block effect on the noedit control (vs b32) ---")
            for cname, ctl in sorted(controls.items()):
                if ctl is ref or ctl["blk"] == ref["blk"]:
                    continue
                cmap = {d["i"]: d for d in ref["det"]}
                tmap = {d["i"]: d for d in ctl["det"]}
                common = sorted(set(cmap) & set(tmap))
                b = c = 0
                for i in common:
                    oc, ot = ok_of(cmap[i]), ok_of(tmap[i])
                    if not oc and ot:
                        b += 1
                    elif oc and not ot:
                        c += 1
                pv = mcnemar_exact(b, c)
                dp = 100 * (ctl["k"] / ctl["n"] - ref["k"] / ref["n"])
                print(f"  {cname:22s} vs noedit_b32   (n={len(common):3d}): "
                      f"diff={dp:+5.1f}pp  fix/brk={b}/{c}  McNemar p={pv:.4f}")

    st = R.get("stats", {})
    if st:
        print(f"\n{'=' * 104}\nIN-KERNEL STATS SECTION (cross-check)\n{'=' * 104}")
        print(json.dumps(st, indent=1, ensure_ascii=False)[:4500])

    print("\n(*) p < 0.05, exact McNemar on paired task outcomes.")


if __name__ == "__main__":
    main()
