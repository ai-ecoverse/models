"""Mistral's 3B vision encoder and embedding graphs, re-weighted for another Ministral 3 checkpoint.

    python resize.py --ref REF --ckpt CKPT --out OUT [--graphs vision_encoder_fp16 embed_tokens_fp16]

Every Ministral 3 model (3B, 8B, 14B) has the same Pixtral vision tower (vision_config is identical); only the
projector's output width and the embedding table follow the text model's hidden size. So the reference's
vision_encoder_fp16.onnx and embed_tokens_fp16.onnx graphs are kept as they are, node for node, and only their
weights change: each initializer gets the checkpoint's tensor (transposed for MatMul weights, cast to the
initializer's dtype), and the projector weights, the embedding table and the hidden-size output dimension take the
checkpoint's shape. Any other shape mismatch is an error.
"""
import argparse, json, os, re, sys
import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shards import save_sharded, set_last_dim, to_numpy_dtype  # noqa: E402

# reference initializer -> checkpoint tensor, transposed (ONNX MatMul weights are (in, out))
VISION = [
    (r"patch_conv\.weight", "vision_tower.patch_conv.weight", False),
    (r"model\.layers\.-1\.ln_pre_layernorm\.weight", "vision_tower.ln_pre.weight", False),
    (r"model\.layers\.(\d+)\.(attention_norm|ffn_norm)_layernorm\.weight", r"vision_tower.transformer.layers.\1.\2.weight", False),
    (r"model\.layers\.(\d+)\.attn\.(q|k|v|o)_proj\.MatMul\.weight", r"vision_tower.transformer.layers.\1.attention.\2_proj.weight", True),
    (r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.MatMul\.weight", r"vision_tower.transformer.layers.\1.feed_forward.\2_proj.weight", True),
    (r"model\.layers\.projector\.multi_modal_projector\.norm_layernorm\.weight", "multi_modal_projector.norm.weight", False),
    (r"model\.multimodal_projector\.patch_merger\.Linear\.weight", "multi_modal_projector.patch_merger.merging_layer.weight", True),
    (r"model\.multimodal_projector\.(linear_1|linear_2)\.MatMul\.weight", r"multi_modal_projector.\1.weight", True),
]
EMBED = [(r"model\.embed_tokens\.weight", "language_model.model.embed_tokens.weight", False)]
RESIZABLE = re.compile(r"multimodal_projector\.linear_[12]\.|embed_tokens")


class Checkpoint:
    def __init__(self, path):
        import torch
        from safetensors import safe_open
        self.torch = torch
        self.index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {f: safe_open(os.path.join(path, f), framework="pt") for f in set(self.index.values())}

    def get(self, mapping, name):
        for pat, repl, transpose in mapping:
            if re.fullmatch(pat, name):
                src = re.sub(pat, repl, name)
                t = self.files[self.index[src]].get_tensor(src).to(self.torch.float32)
                return np.ascontiguousarray((t.T if transpose else t).numpy())
        raise SystemExit(f"no checkpoint tensor for {name}")


def reweight(ref_path, ckpt, mapping, hidden):
    model = onnx.load(ref_path, load_external_data=False)
    changed = []
    for t in model.graph.initializer:
        if not t.external_data:
            continue
        w = ckpt.get(mapping, t.name)
        if list(w.shape) != list(t.dims):
            if not RESIZABLE.search(t.name):
                raise SystemExit(f"{t.name}: checkpoint shape {list(w.shape)} != reference {list(t.dims)}")
            changed.append((t.name, list(t.dims), list(w.shape)))
            del t.dims[:]
            t.dims.extend(w.shape)
        del t.external_data[:]
        t.data_location = onnx.TensorProto.DEFAULT
        t.raw_data = w.astype(to_numpy_dtype(t.data_type)).tobytes()
    for o in model.graph.output:
        set_last_dim(o, hidden)
    # Intermediate shapes were inferred for the reference's width; ONNX Runtime re-infers them.
    del model.graph.value_info[:]
    return model, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference repo with onnx/*.onnx graph files")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--graphs", nargs="+", default=["vision_encoder_fp16", "embed_tokens_fp16"])
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    hidden = cfg["text_config"]["hidden_size"]
    ref_vision = json.load(open(os.path.join(a.ref, "config.json")))["vision_config"]
    strip = lambda c: {k: v for k, v in c.items() if k not in ("model_type",)}
    if strip(ref_vision) != strip(cfg["vision_config"]):
        raise SystemExit("vision_config differs from the reference's; the vision graph can't be reused")
    ckpt = Checkpoint(a.ckpt)
    report = {}
    for g in a.graphs:
        mapping = VISION if g.startswith("vision_encoder") else EMBED
        model, changed = reweight(os.path.join(a.ref, "onnx", f"{g}.onnx"), ckpt, mapping, hidden)
        files = save_sharded(model, os.path.join(a.out, "onnx"), f"{g}.onnx")
        report[f"{g}.onnx"] = {"data_files": files, "resized": changed}
        print(g, json.dumps(report[f"{g}.onnx"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
