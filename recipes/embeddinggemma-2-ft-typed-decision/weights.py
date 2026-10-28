"""Write a fine-tuned EmbeddingGemma 2 text encoder's weights into onnx-community's text graph.

onnx-community/embeddinggemma-2-ONNX `onnx/model.onnx` is the text model: `input_ids`, `attention_mask` and three
media inputs in, `last_hidden_state [B, T, 768]` and `sentence_embedding [B, 768]` (masked mean of
`last_hidden_state`, L2-normalized) out. The graph file is copied unchanged. Its external data file is written anew at
the reference's size: every initializer keeps its location, offset, length, dtype and shape, and its bytes come from
the source checkpoint, laid out the way the reference's are:

  embed_tokens               the table times float32(sqrt(hidden_size)), the scale EmbeddingGemma 2 applies per lookup
  *.MatMul.weight            Linear weight, transposed ([out, in] -> [in, out])
  attn.qkv_proj              q, k and v projections, concatenated on the output axis
  attn.qkv_norm              q_norm tiled over the query heads, k_norm over the key heads, ones for v (v_norm has no scale)
  mlp.gate_up_proj           gate and up projections, concatenated
  layer_scalar (layers 0-22) the layer's scalar; the last one is dropped by the reference, since the final RMSNorm
                             cancels it (up to its epsilon)
  rotary inv_freq, ones_*    copied from the reference (config only)

    python weights.py --ref REF --ckpt SAFETENSORS --prefix encoder.language_model. --out OUT_DIR
    python weights.py --ref REF --ckpt BASE.safetensors --prefix model.language_model. --check   # self-test

--check builds from the base model the reference was converted from and compares every tensor's bytes with the
reference's own: the mapping is then the one onnx-community used, so the fine-tune lands where the graph expects it.
"""
import argparse, json, os, re, shutil, sys
import numpy as np
import onnx
from onnx import numpy_helper
from safetensors import safe_open

GRAPH = "onnx/model.onnx"
COPY = {"model.rotary_emb.full_attention_inv_freq", "model.rotary_emb.sliding_attention_inv_freq",
        "model.ones_256", "model.ones_512"}


def ext(t):
    d = {e.key: e.value for e in t.external_data}
    return d["location"], int(d.get("offset", 0)), int(d["length"])


class Ckpt:
    def __init__(self, path, prefix):
        self.f = safe_open(path, "np")
        self.p = prefix
        self.used = set()

    def __call__(self, name):
        self.used.add(self.p + name)
        return self.f.get_tensor(self.p + name).astype(np.float32)

    def unused(self):
        return sorted(k for k in self.f.keys() if k.startswith(self.p) and k not in self.used)


def build(name, w, text_cfg):
    """The reference initializer `name` computed from checkpoint accessor `w`, or None for COPY tensors."""
    if name in COPY:
        return None
    m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        return {
            # EmbeddingGemma2TextScaledWordEmbedding's sqrt(hidden_size), folded in as float32 (bit-exact per lookup)
            "model.embed_tokens.weight": lambda: w("embed_tokens.weight") * np.float32(text_cfg["hidden_size"] ** 0.5),
            "model.ple.per_layer_model_projection.MatMul.weight": lambda: w("ple.per_layer_model_projection.weight").T,
            "model.ple.per_layer_projection_norm.weight": lambda: w("ple.per_layer_projection_norm.weight"),
            "model.norm.weight": lambda: w("norm.weight"),
            "model.embedding_projection.MatMul.weight": lambda: w("embedding_projection.weight").T,
        }[name]()
    i, rest = int(m[1]), m[2]
    L = f"layers.{i}."
    pl = text_cfg.get("per_layer_config", {}).get(f"{i:02d}", {})
    hd = pl.get("head_dim", text_cfg["head_dim"])
    kv = pl.get("num_key_value_heads", text_cfg["num_key_value_heads"])
    heads = text_cfg["num_attention_heads"]
    simple = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
        "ple.post_per_layer_input_norm.weight": "ple_block.post_per_layer_input_norm.weight",
        "layer_scalar": "layer_scalar",
    }
    linear = {
        "attn.o_proj.MatMul.weight": "self_attn.o_proj.weight",
        "mlp.down_proj.MatMul.weight": "mlp.down_proj.weight",
        "ple.per_layer_input_gate.MatMul.weight": "ple_block.per_layer_input_gate.weight",
        "ple.per_layer_projection.MatMul.weight": "ple_block.per_layer_projection.weight",
    }
    if rest in simple:
        return w(L + simple[rest])
    if rest in linear:
        return w(L + linear[rest]).T
    if rest == "attn.qkv_proj.MatMul.weight":
        return np.concatenate([w(L + f"self_attn.{p}_proj.weight") for p in "qkv"], 0).T
    if rest == "attn.qkv_norm.weight":
        return np.concatenate([np.tile(w(L + "self_attn.q_norm.weight"), heads),
                               np.tile(w(L + "self_attn.k_norm.weight"), kv), np.ones(kv * hd, np.float32)])
    if rest == "mlp.gate_up_proj.MatMul.weight":
        return np.concatenate([w(L + "mlp.gate_proj.weight"), w(L + "mlp.up_proj.weight")], 0).T
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="onnx-community/embeddinggemma-2-ONNX snapshot")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prefix", required=True, help="name prefix of the text model in the checkpoint")
    ap.add_argument("--config", help="config.json holding text_config (default: next to --ckpt)")
    ap.add_argument("--out", help="write OUT/onnx/model.onnx{,_data}")
    ap.add_argument("--check", action="store_true", help="compare with the reference's bytes instead of writing")
    ap.add_argument("--report")
    a = ap.parse_args()
    cfg = json.load(open(a.config or os.path.join(os.path.dirname(a.ckpt), "config.json")))
    text_cfg = (cfg.get("encoder_config") or cfg)["text_config"]
    w = Ckpt(a.ckpt, a.prefix)
    graph = onnx.load(os.path.join(a.ref, GRAPH), load_external_data=False)
    data, inline = {}, []
    for t in graph.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            if not t.name.startswith("/model/constants/") and t.name not in COPY:
                inline.append(t)
            continue
        if t.data_type != onnx.TensorProto.FLOAT:
            raise SystemExit(f"{t.name}: expected float32, got {t.data_type}")
        loc, off, n = ext(t)
        data.setdefault(loc, []).append((t, off, n))
    report = {"tensors": 0, "copied": [], "mismatched": [], "max_abs_diff": 0.0, "inline_changed": []}

    def compare(name, r, arr):
        if not np.array_equal(r, arr):
            d = float(np.abs(r - arr).max())
            report["mismatched"].append({"name": name, "max_abs_diff": d})
            report["max_abs_diff"] = max(report["max_abs_diff"], d)

    for t in inline:   # small tensors stored in the graph file (the layer scalars)
        r = numpy_helper.to_array(t)
        arr = np.ascontiguousarray(build(t.name, w, text_cfg), np.float32).reshape(r.shape)
        report["tensors"] += 1
        if a.check:
            compare(t.name, r, arr)
        elif not np.array_equal(r, arr):
            t.CopyFrom(numpy_helper.from_array(arr, t.name))
            report["inline_changed"].append(t.name)
    for loc, items in data.items():
        src = os.path.join(a.ref, "onnx", loc)
        size = os.path.getsize(src)
        if a.out:
            os.makedirs(os.path.join(a.out, "onnx"), exist_ok=True)
            dst_path = os.path.join(a.out, "onnx", loc)
            shutil.copyfile(src, dst_path)   # same size; every mapped tensor is overwritten below
            dst = open(dst_path, "r+b")
        with open(src, "rb") as ref:
            for t, off, n in sorted(items, key=lambda x: x[1]):
                arr = build(t.name, w, text_cfg)
                report["tensors"] += 1
                if arr is None:
                    report["copied"].append(t.name)
                    continue
                arr = np.ascontiguousarray(arr, np.float32)
                if list(arr.shape) != list(t.dims) or arr.nbytes != n:
                    raise SystemExit(f"{t.name}: built {arr.shape}, reference {list(t.dims)}")
                if a.check:
                    ref.seek(off)
                    compare(t.name, np.frombuffer(ref.read(n), np.float32).reshape(arr.shape), arr)
                if a.out:
                    dst.seek(off)
                    dst.write(arr.tobytes())
        if a.out:
            dst.close()
            if os.path.getsize(dst_path) != size:
                raise SystemExit(f"{loc}: size changed")
    if a.out and report["inline_changed"]:
        onnx.save_model(graph, os.path.join(a.out, GRAPH))   # external data references stay as they are
    elif a.out:
        shutil.copyfile(os.path.join(a.ref, GRAPH), os.path.join(a.out, GRAPH))
    last = f"{a.prefix}layers.{text_cfg['num_hidden_layers'] - 1}.layer_scalar"   # dropped by the reference
    unused = [k for k in w.unused() if k != last]
    report["unused_checkpoint_tensors"] = unused
    print(json.dumps({k: v if not isinstance(v, list) or len(v) < 8 else f"{len(v)} items" for k, v in report.items()}))
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    if report["mismatched"] or unused:
        raise SystemExit(f"{len(report['mismatched'])} tensors differ from the reference, unused: {unused[:5]}")


if __name__ == "__main__":
    sys.exit(main())
