"""Step-by-step debug: stock generate vs generate_wet_ink(revision_blocks=0)
with an EOS probe — find where the trajectories diverge."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from make_tiny_repo import main as make_repo, DST as REPO  # noqa: E402

import torch  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402

from wet_ink import generate_wet_ink, get_stock_helpers  # noqa: E402

torch.manual_seed(0)

make_repo()
config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True).eval()
torch.manual_seed(0)
mask_id = model.mask_token_id
prompt = torch.randint(200, 2000, (1, 16))

get_transfer_index, get_num, crop = get_stock_helpers(model)
import importlib  # noqa: E402

mod = importlib.import_module(type(model).__module__)

LOG = []


def spy(logits, temperature, mask_index, x, num_transfer_tokens, threshold=None):
    x0, transfer = get_transfer_index(
        logits, temperature, mask_index, x, num_transfer_tokens, threshold
    )
    LOG.append(
        dict(
            n_mask=int(mask_index.sum()),
            committed=[(int(p), int(x0[0, p].item())) for p in transfer.nonzero()[:, 1].tolist()],
        )
    )
    return x0, transfer


mod._get_transfer_index = spy

gen_kwargs = dict(max_new_tokens=64, block_length=32, threshold=0.9)

# reference run without EOS to pick the probe token
out_stock, _ = model.generate(prompt, **gen_kwargs)
eos_probe = int(out_stock[0, 20].item())
print("eos_probe =", eos_probe)

print("\n=== STOCK with eos ===")
LOG.clear()
out_s, nfe_s = model.generate(prompt, eos_token_id=eos_probe, **gen_kwargs)
print("nfe:", nfe_s, "len:", out_s.shape[1])
for r in LOG:
    print("  ", r)

print("\n=== WET-INK W=0 with eos ===")
LOG.clear()
out_w, nfe_w = generate_wet_ink(model, prompt, revision_blocks=0, eos_token_id=eos_probe, **gen_kwargs)
print("nfe:", nfe_w, "len:", out_w.shape[1])
for r in LOG:
    print("  ", r)

print("\nequal outputs:", torch.equal(out_s, out_w))
