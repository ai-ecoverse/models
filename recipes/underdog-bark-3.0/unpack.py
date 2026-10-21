"""Unpack model.bark with the source repo's own bark_loader.py into a transformers ParakeetForTDT folder.

    python unpack.py --src src --processor proc --out hf
"""
import argparse, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--processor", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.src))
import torch  # noqa: E402
from bark_loader import load_state_dict  # noqa: E402  (the source's exact decoder)
from transformers import GenerationConfig, ParakeetForTDT, ParakeetTDTConfig  # noqa: E402

sd, cfg = load_state_dict(f"{a.src}/model.bark")
model = ParakeetForTDT(ParakeetTDTConfig.from_dict(cfg))
missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
assert not unexpected and all("num_batches_tracked" in m for m in missing), (missing[:5], unexpected[:5])
model.generation_config = GenerationConfig.from_pretrained(a.processor)
model.eval().save_pretrained(a.out)
for f in ("processor_config.json", "tokenizer.json", "tokenizer_config.json"):
    shutil.copy(f"{a.processor}/{f}", f"{a.out}/{f}")
print(json.dumps({"tensors": len(sd), "params": sum(v.numel() for v in sd.values())}))
