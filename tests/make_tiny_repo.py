"""Build a tiny local HF repo of Nemotron-Labs-Diffusion (random weights) for
CPU smoke tests. Uses the EXACT architecture files shipped with the 3B model
(verbatim copies from nvidia/Nemotron-Labs-Diffusion-3B, Apache-2.0, see
tests/vendor/NOTICE.md) with a tiny config."""

import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "vendor")
DST = os.path.join(ROOT, "local_model_repo")

FILES = {
    "modeling_nemotron_labs_diffusion.py": "modeling_nemotron_labs_diffusion.py",
    "modeling_ministral.py": "modeling_ministral.py",
    "configuration_nemotron_labs_diffusion.py": "configuration_nemotron_labs_diffusion.py",
}

TINY_CONFIG = {
    "architectures": ["NemotronLabsDiffusionModel"],
    "ar_loss_weight": 1.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_implementation": "sdpa",
    "auto_map": {
        "AutoConfig": "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        "AutoModel": "modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel",
    },
    "block_size": 32,
    "bos_token_id": 1,
    "dlm_loss_weight": None,
    "dlm_paradigm": "bidirectional",
    "dp_varying_mask_ratio": False,
    "eos_token_id": 11,
    "head_dim": 32,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 704,
    "mask_token_id": 100,
    "max_position_embeddings": 2048,
    "mlp_bias": False,
    "model_type": "nemotron_labs_diffusion",
    "num_attention_heads": 8,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 4.0,
        "llama_4_scaling_beta": 0.1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "rope_type": "yarn",
    },
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "5.0.0",
    "use_cache": False,
    "vocab_size": 2048,
}


def main():
    os.makedirs(DST, exist_ok=True)
    for name, src_name in FILES.items():
        shutil.copyfile(os.path.join(SRC, src_name), os.path.join(DST, name))
    with open(os.path.join(DST, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f, indent=2)
    with open(os.path.join(DST, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": 11, "use_cache": False}, f)
    print("tiny repo ready at", DST)


if __name__ == "__main__":
    main()
