"""Summarize results.json from the Kaggle kernel into a compact table
(Russian) for the README + print mechanism statistics."""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
with open(path) as f:
    R = json.load(f)

print("META:", R.get("meta", {}))
print()
print("EQUIVALENCE (W=0 vs stock):", R.get("equivalence", {}))
print()
print("SANITY:")
for k, v in R.get("sanity", {}).items():
    print(f"  {k}: nfe={v.get('nfe')} text={v.get('text', '')[:120]!r}")
print()
print("QUALITATIVE:")
for q in R.get("qualitative", []):
    s = q.get("stock", {})
    w = q.get("wet_ink_w1", {})
    print(f"  Q: {q['question'][:70]!r}")
    print(f"    stock  nfe={s.get('nfe')}: {s.get('text', '')[:100]!r}")
    print(f"    wet_w1 nfe={w.get('nfe')}: {w.get('text', '')[:100]!r}")
    print(f"    remask={w.get('remask_events')} revised={w.get('revision_token_changes')} "
          f"changed_vs_stock={q.get('n_tokens_changed_vs_stock')}")
print()
hdr = f"{'config':<16}{'acc%':>7}{'avg_tok':>9}{'avg_nfe':>9}{'TPF':>7}{'remask':>8}{'revised':>9}{'tail':>7}{'cap':>5}{'wall_s':>8}"
print("GSM8K (protocol of the official evaluate.py):")
print(hdr)
for name, r in R.get("gsm8k", {}).items():
    if "skipped" in r:
        print(f"{name:<16} SKIPPED ({r['skipped']})")
        continue
    print(f"{name:<16}{r['accuracy']:>7.2f}{r['avg_tokens']:>9.1f}{r['avg_nfe']:>9.1f}"
          f"{r['tpf']:>7.2f}{r['remask_events']:>8}{r['revision_token_changes']:>9}"
          f"{r['safety_tail_steps']:>7}{r.get('hard_cap_hits', 0):>5}{r['wall_seconds']:>8.0f}")

# README-style markdown table
print()
print("| Конфиг | Acc% | avg токенов | avg NFE | TPF | ремасков | ревизий токенов |")
print("|---|---|---|---|---|---|---|")
for name, r in R.get("gsm8k", {}).items():
    if "skipped" in r:
        print(f"| {name} | пропущен ({r['skipped']}) | | | | | |")
        continue
    print(f"| {name} | {r['accuracy']:.2f} | {r['avg_tokens']:.1f} | {r['avg_nfe']:.1f} "
          f"| {r['tpf']:.2f} | {r['remask_events']} | {r['revision_token_changes']} |")
