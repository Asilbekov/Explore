#!/usr/bin/env python3
"""Analyze wet-ink v5 results vs the v4b baselines (cross-run pairing).

Pairs OUR wet-ink decoder arms (results_v5.json) with the stock/noedit
baselines from the completed v4b run (results_v4.json) — same model
(LLaDA2.2-mini), same 4-bit NF4 quantization, same prompts (GSM8K test
order / HumanEval order), same gen_length=160, greedy — so task-index
pairing is valid. Also pairs within v5 vs the frontier-only control.

Reports: accuracy + Wilson 95% CI, exact McNemar, NFE ratio (the honest
cost of the one-token frontier), churn, rewrites-per-task, truncation.

Usage: python3 scripts/analyze_v5.py [results_v5.json] [results_v4.json]
"""
import json
import math
import sys
from math import comb

P5 = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/wet_ink/results/results_v5.json"
P4 = sys.argv[2] if len(sys.argv) > 2 else "/home/z/my-project/wet_ink/results/results_v4.json"
TOK_CAP = 160


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
    v = d.get("ok", d.get("passed"))
    return bool(v)


def arm_rows(R, bench):
    rows = {}
    for name, a in R.get(bench, {}).items():
        if "skipped" in a or "error" in a:
            continue
        det = [d for d in a.get("details", []) if "error" not in d]
        if not det:
            continue
        n = len(det)
        k = sum(ok_of(d) for d in det)
        p, lo, hi = wilson(k, n)
        rows[name] = dict(
            det=det, n=n, k=k, acc=100 * p, lo=100 * lo, hi=100 * hi,
            nfe=sum(d.get("nfe", 0) for d in det) / n,
            churn=sum(d.get("churn", 0) for d in det) / n,
            tok=sum(d.get("n_tokens", 0) for d in det) / n,
            rewrites=sum(d.get("rewrites") or 0 for d in det) / n,
            opened=sum(d.get("opened") or 0 for d in det) / n,
            wall=a.get("wall_seconds", a.get("gen_wall_seconds", 0)),
            cfg={kk: a.get(kk) for kk in ("mode", "window", "block_length",
                                          "editing_threshold", "max_post_steps")},
        )
    return rows


def pair(base_row, arm_row, base_name, arm_name):
    b_map = {d["i"]: d for d in base_row["det"]}
    a_map = {d["i"]: d for d in arm_row["det"]}
    common = sorted(set(b_map) & set(a_map))
    b = c = 0
    for i in common:
        ob, oa = ok_of(b_map[i]), ok_of(a_map[i])
        if not ob and oa:
            b += 1
        elif ob and not oa:
            c += 1
    nfe_a = sum(a_map[i].get("nfe", 0) for i in common) / max(len(common), 1)
    nfe_b = sum(b_map[i].get("nfe", 0) for i in common) / max(len(common), 1)
    return dict(
        n_common=len(common), acc=arm_row["acc"], ci95=[arm_row["lo"], arm_row["hi"]],
        base_acc=base_row["acc"], diff=arm_row["acc"] - base_row["acc"],
        fixed=b, broken=c, mcnemar_p=mcnemar_exact(b, c),
        nfe_ratio=nfe_a / nfe_b if nfe_b else float("nan"),
        nfe_base=nfe_b, nfe_arm=nfe_a,
    )


def main():
    R5 = json.load(open(P5))
    try:
        R4 = json.load(open(P4))
    except FileNotFoundError:
        print(f"!! v4b baselines not found at {P4} — within-v5 pairing only")
        R4 = {}

    meta = R5.get("meta", {})
    print("=" * 104)
    print("WET-INK v5 — OUR DECODER (one-token frontier + sliding trained-T2T window)")
    print("model:", meta.get("model"), "| quant:", meta.get("quantization"))
    print("idea:", meta.get("idea"))
    print("=" * 104)

    san = R5.get("sanity", {})
    for k, v in san.items():
        wv = v.get("wetink") or {}
        print(f"sanity {k}: nfe={v.get('nfe')} churn={v.get('churn')} "
              f"wall={v.get('wall_seconds', 0):.0f}s "
              + (f"steps={wv.get('steps')} opened={wv.get('opened')} "
                 f"rewrites={wv.get('rewrites')}" if wv else ""))
        print(f"  out[:140]: {v.get('output_prefix', '')[:140]!r}")
    for k, v in R5.get("equivalence", {}).items():
        print(f"equivalence {k}: ids_equal={v.get('ids_equal')} "
              f"nfe {v.get('nfe1')}=={v.get('nfe2')}")

    for bench in ("gsm8k", "humaneval"):
        rows5 = arm_rows(R5, bench)
        rows4 = arm_rows(R4, bench)
        if not rows5:
            continue
        print(f"\n{'=' * 104}\n{bench.upper()} — v5 arms\n{'=' * 104}")
        for name, r in rows5.items():
            print(f"  {name:22s} n={r['n']:3d}  acc={r['acc']:5.1f}% "
                  f"CI[{r['lo']:4.1f},{r['hi']:4.1f}]  nfe={r['nfe']:6.1f} "
                  f"churn={r['churn']:5.2f}  rewrites/task={r['rewrites']:5.1f} "
                  f"tok={r['tok']:5.1f}  W={r['cfg'].get('window')} "
                  f"et={r['cfg'].get('editing_threshold')}  wall={r['wall']:6.0f}s")

        # --- vs v4b baselines (cross-run, same tasks by index) ---
        bases = {b: rows4[b] for b in ("stock_b32", "noedit_b32") if b in rows4}
        if bases:
            print(f"\n  --- vs v4b baselines (paired by task index) ---")
            for arm_name, r in rows5.items():
                for bname, brow in bases.items():
                    s = pair(brow, r, bname, arm_name)
                    star = " *" if s["mcnemar_p"] < 0.05 else ""
                    print(f"  {arm_name:22s} vs {bname:12s} (n={s['n_common']:3d}): "
                          f"diff={s['diff']:+6.1f}pp  win/loss={s['fixed']}/{s['broken']}  "
                          f"McNemar p={s['mcnemar_p']:.4f}{star}  "
                          f"NFE x{s['nfe_ratio']:.2f} ({s['nfe_base']:.0f}->{s['nfe_arm']:.0f})")

        # --- vs within-v5 frontier-only control ---
        ctrl = rows5.get("wetink_noedit_w16")
        if ctrl:
            print(f"\n  --- vs frontier-only control (wetink_noedit_w16) ---")
            for arm_name, r in rows5.items():
                if arm_name == "wetink_noedit_w16":
                    continue
                s = pair(ctrl, r, "wetink_noedit_w16", arm_name)
                star = " *" if s["mcnemar_p"] < 0.05 else ""
                print(f"  {arm_name:22s} vs control     (n={s['n_common']:3d}): "
                      f"diff={s['diff']:+6.1f}pp  win/loss={s['fixed']}/{s['broken']}  "
                      f"McNemar p={s['mcnemar_p']:.4f}{star}  "
                      f"NFE x{s['nfe_ratio']:.2f}")

        # --- window dose-response ---
        ws = [rows5[w] for w in ("wetink_w8", "wetink_w16", "wetink_w32") if w in rows5]
        if len(ws) >= 2:
            print(f"\n  --- window dose-response (accuracy / rewrites per task) ---")
            for name in ("wetink_w8", "wetink_w16", "wetink_w32"):
                if name in rows5:
                    r = rows5[name]
                    print(f"  W={r['cfg'].get('window'):>3}: acc={r['acc']:5.1f}% "
                          f"CI[{r['lo']:4.1f},{r['hi']:4.1f}]  rewrites/task={r['rewrites']:5.1f}")

    print("\n(*) p < 0.05, exact McNemar on paired task outcomes.")
    print("NFE ratio > 1 = the honest price of the one-token frontier.")


if __name__ == "__main__":
    main()
