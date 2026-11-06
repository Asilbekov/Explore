"""Long-block smoke test (v3 suite de-risk): b64 / b128 on the tiny model.

The v2 test suite validated the window/revision machinery at block 32 only.
The v3 benchmark runs blocks 64 and 128 — this test exercises exactly that
plumbing before GPU time is spent:

  1. stock generate() at b64 and b128;
  2. EQUIVALENCE: W=0 bit-identical to stock at the same block size
     (reme_di and codr policies);
  3. reme_di W=1 at b64 (s64) and b128 (s128): terminates with no [MASK]
     left, exactly denoise_steps forwards per block, revision happens;
  4. codr W=1 at b64 and b128: terminates, no [MASK] left, bounded NFE;
  5. NFE accounting sanity: wet NFE >= stock NFE at same block size.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from make_tiny_repo import main as make_repo, DST as REPO  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModel, AutoConfig  # noqa: E402

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

    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True).eval()
    mask_id = model.mask_token_id
    prompt = torch.randint(200, 2000, (1, 16))

    for bl, steps in ((64, 64), (128, 128)):
        mnt = 4 * bl  # multiple of the block size
        gen_kwargs = dict(max_new_tokens=mnt, block_length=bl, threshold=0.9)

        out_stock, nfe_stock = model.generate(prompt, **gen_kwargs)
        check(f"b{bl}: stock generate runs",
              out_stock.shape == (1, 16 + mnt), f"nfe={nfe_stock}")
        check(f"b{bl}: stock no [MASK]", not bool((out_stock == mask_id).any()))

        for pol in ("reme_di", "codr"):
            out0, nfe0 = generate_wet_ink(
                model, prompt, revision_blocks=0, remask_policy=pol, **gen_kwargs
            )
            check(f"b{bl}: W=0 {pol} identical ids", bool(torch.equal(out_stock, out0)))
            check(f"b{bl}: W=0 {pol} identical nfe", nfe_stock == nfe0,
                  f"{nfe_stock} vs {nfe0}")

        # reme_di at matched density
        out_r, nfe_r, st_r = generate_wet_ink(
            model, prompt, revision_blocks=1, remask_policy="reme_di",
            denoise_steps=steps, return_stats=True, **gen_kwargs
        )
        check(f"b{bl}: reme_di W=1 terminates clean",
              out_r.shape == (1, 16 + mnt) and not bool((out_r == mask_id).any()))
        per_block = st_r["per_block"]
        check(f"b{bl}: reme_di per-block stats present",
              len(per_block) == mnt // bl and all(
                  {"block", "window_len", "remasked", "revised_tokens"} <= set(pb)
                  for pb in per_block))
        # NFE is deterministic on the tiny model (EOS never fires on random
        # weights): blocks x (denoise_steps + 1 post-block causal forward)
        expected_nfe = (mnt // bl) * (steps + 1)
        check(f"b{bl}: reme_di NFE == blocks x (steps+1)",
              nfe_r == expected_nfe, f"{nfe_r} vs {expected_nfe}")
        check(f"b{bl}: reme_di revision happens",
              st_r["revision_token_changes"] > 0,
              f"changes={st_r['revision_token_changes']}")
        check(f"b{bl}: reme_di no safety tail",
              st_r.get("safety_tail_steps", 0) == 0 and st_r.get("hard_cap_hits", 0) == 0)

        # codr
        out_c, nfe_c, st_c = generate_wet_ink(
            model, prompt, revision_blocks=1, remask_policy="codr",
            return_stats=True, **gen_kwargs
        )
        check(f"b{bl}: codr W=1 terminates clean",
              out_c.shape == (1, 16 + mnt) and not bool((out_c == mask_id).any()))
        check(f"b{bl}: codr bounded NFE",
              nfe_stock <= nfe_c <= 4 * nfe_stock + 4 * (2 * bl),
              f"stock={nfe_stock} codr={nfe_c}")
        check(f"b{bl}: wet NFE >= stock NFE", nfe_c >= nfe_stock and nfe_r >= nfe_stock,
              f"stock={nfe_stock} codr={nfe_c} remedi={nfe_r}")

    print()
    if FAILURES:
        print("FAILED:", FAILURES)
        sys.exit(1)
    print("LONG-BLOCK SMOKE TEST PASSED (b64, b128)")


if __name__ == "__main__":
    main()
