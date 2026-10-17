"""Rebuild Mistral's Ministral-3-3B-Instruct-2512-ONNX graphs with another Ministral 3 3B checkpoint's weights.

The .onnx graph files are copied unchanged. Each external-data file is written anew at the reference's size: every
initializer keeps its location, offset, length, dtype and shape, and its bytes come from the source weights, cast or
quantized exactly the way the reference was:

  plain fp32 / fp16           cast (bf16 -> fp32 -> fp16)
  *_Q4 + *_scales             symmetric 4-bit, block 32: onnxruntime quantize_matmul_4bits (C++)
  *_quant + *_scales + *_zp   asymmetric 4/8-bit, block 32: onnxruntime's numpy quant_slice_asymmetric with
                              np.round, computed in the scales' dtype (fp32, or fp16 for the f16 variants)
  cos_cache / sin_cache       copied from the reference (they depend on the config only)

    python build.py --ref REF --out OUT --variants q4f16 fp16 q4 quantized fp32 \
        (--ckpt CKPT | --src-onnx REF)

REF holds the reference repo (onnx/*.onnx, and onnx/decoder_model_merged.onnx_data for the rope caches; without it
they are range-read from the Hub). --ckpt reads a Mistral3ForConditionalGeneration safetensors checkpoint.
--src-onnx reads the fp32 weights from the reference's own fp32 graphs instead: building every other variant from
it must reproduce the reference byte for byte (the self-test, see selftest.py).
"""
import argparse, json, os, re, shutil, sys, time
import numpy as np
import onnx
from onnxruntime.capi._pybind_state import quantize_matmul_4bits

SUFFIX = {"fp32": "", "fp16": "_fp16", "q4": "_q4", "q4f16": "_q4f16", "quantized": "_quantized"}
GRAPHS = ["decoder_model_merged", "embed_tokens", "vision_encoder"]
NP = {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16, onnx.TensorProto.UINT8: np.uint8}
REF_REPO = "mistralai/Ministral-3-3B-Instruct-2512-ONNX"
REF_REV = "e1128d389145f96bb1c3c2625cb8ca255ffa7034"

DECODER = [
    (r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight", r"language_model.model.layers.\1.\2.weight", False),
    (r"model\.layers\.(\d+)\.attn\.(q|k|v|o)_proj\.MatMul\.weight", r"language_model.model.layers.\1.self_attn.\2_proj.weight", True),
    (r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.MatMul\.weight", r"language_model.model.layers.\1.mlp.\2_proj.weight", True),
    (r"model\.layers\.26\.final_norm_layernorm\.weight", "language_model.model.norm.weight", False),
    (r"lm_head\.MatMul\.weight", "language_model.model.embed_tokens.weight", True),
    (r"model\.embed_tokens\.weight", "language_model.model.embed_tokens.weight", False),
]
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


def ext(t):
    d = {e.key: e.value for e in t.external_data}
    return d["location"], int(d["offset"]), int(d["length"])


def read_ext(base, t):
    loc, off, n = ext(t)
    with open(os.path.join(base, loc), "rb") as f:
        f.seek(off)
        b = f.read(n)
    if len(b) != n:
        raise SystemExit(f"{t.name}: short read from {loc}")
    return np.frombuffer(b, NP[t.data_type]).reshape(list(t.dims))


def hub_range(path, off, n, chunk=16 << 20):
    import requests
    from huggingface_hub import hf_hub_url
    url, out = hf_hub_url(REF_REPO, path, revision=REF_REV), bytearray()
    with requests.Session() as s:
        while len(out) < n:
            a = off + len(out)
            b = min(off + n, a + chunk) - 1
            for _ in range(10):
                try:
                    r = s.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                    if r.status_code == 206 and len(r.content) == b - a + 1:
                        break
                except requests.RequestException:
                    time.sleep(3)
            else:
                raise SystemExit(f"range read of {path} failed")
            out += r.content
    return bytes(out)


class Checkpoint:
    def __init__(self, path):
        import torch
        from safetensors import safe_open
        self.torch = torch
        index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {f: safe_open(os.path.join(path, f), framework="pt") for f in set(index.values())}
        self.index = index

    def get(self, graph, name):
        for pat, repl, transpose in VISION if graph == "vision_encoder" else DECODER:
            if re.fullmatch(pat, name):
                src = re.sub(pat, repl, name)
                t = self.files[self.index[src]].get_tensor(src).to(self.torch.float32)
                return np.ascontiguousarray((t.T if transpose else t).numpy())
        raise SystemExit(f"{graph}: no checkpoint tensor for {name}")


class OnnxSource:
    def __init__(self, ref):
        self.ref = ref
        self.inits = {}
        for g in GRAPHS:
            m = onnx.load(os.path.join(ref, "onnx", g + ".onnx"), load_external_data=False)
            self.inits[g] = {t.name: t for t in m.graph.initializer}

    def get(self, graph, name):
        t = self.inits[graph].get(name)
        if t is None:
            raise SystemExit(f"{graph}: no fp32 reference tensor for {name}")
        return np.array(read_ext(os.path.join(self.ref, "onnx"), t), np.float32)


def sym4(w):
    k, n = w.shape
    kb = (k + 31) // 32
    packed = np.zeros((n, kb, 16), np.uint8)
    zp = np.zeros((n, (kb + 1) // 2), np.uint8)
    scales = np.zeros((n, kb), w.dtype)
    quantize_matmul_4bits(packed, np.ascontiguousarray(w), scales, zp, 32, n, k, True)
    return packed, scales


def asym(w, bits, rows=8192):
    """onnxruntime's quant_slice_asymmetric over blocks of 32 along K, in w's dtype, for w of shape (K, N)."""
    k, n = w.shape
    if k % 32:
        raise SystemExit("K must be a multiple of 32")
    levels = (1 << bits) - 1
    mid = (levels + 1) // 2
    dt = w.dtype.type
    kb = k // 32
    q = np.empty((n, kb, 32), np.uint8)
    s = np.empty((n, kb), w.dtype)
    z = np.empty((n, kb), np.uint8)
    wt = w.T
    for r0 in range(0, n, rows):
        x = np.ascontiguousarray(wt[r0:r0 + rows]).reshape(-1, kb, 32)
        mn = np.minimum(x.min(-1, keepdims=True), dt(0))
        mx = np.maximum(x.max(-1, keepdims=True), dt(0))
        sc = (mx - mn) / dt(levels)
        with np.errstate(divide="ignore", invalid="ignore"):
            zp = np.where(sc == 0, mid, -mn / sc).round().clip(0, levels).astype(np.uint8)
            qv = np.where(sc == 0, mid, x / sc + zp).round().clip(0, levels).astype(np.uint8)
        q[r0:r0 + rows], s[r0:r0 + rows], z[r0:r0 + rows] = qv, sc[..., 0], zp[..., 0]
    if bits == 8:
        return q, s, z
    qf = q.reshape(n, -1)
    packed = (qf[:, 0::2] & 0xF) | ((qf[:, 1::2] & 0xF) << 4)
    if kb % 2:
        z = np.concatenate([z, np.zeros((n, 1), np.uint8)], 1)
    zpk = (z[:, 0::2] & 0xF) | ((z[:, 1::2] & 0xF) << 4)
    return packed.reshape(n, kb, 16), s, zpk


def plan(graph, model):
    inits = {t.name: t for t in model.graph.initializer}
    jobs, used = [], set()
    for name, t in inits.items():
        if name.endswith("_Q4"):
            base = name[:-3]
            jobs.append(("sym4", base, {"q": t, "s": inits[base + "_scales"]}))
            used |= {name, base + "_scales"}
        elif name.endswith("_quant"):
            base = name[:-6]
            jobs.append(("asym", base, {"q": t, "s": inits[base + "_scales"], "z": inits[base + "_zp"]}))
            used |= {name, base + "_scales", base + "_zp"}
    for name, t in inits.items():
        if name not in used and t.external_data:
            jobs.append(("plain", name, {"t": t}))
    covered = {p.name for _, _, parts in jobs for p in parts.values()}
    missing = {t.name for t in model.graph.initializer if t.external_data} - covered
    if missing:
        raise SystemExit(f"{graph}: initializers without a job: {sorted(missing)[:5]}")
    return jobs


def fp32_name(graph, base, names):
    if base in names:
        return base
    for cand in names:
        if cand.replace(".", "_") == base:
            return cand
    raise SystemExit(f"{graph}: no fp32 initializer for {base}")


def rope_caches(ref):
    m = onnx.load(os.path.join(ref, "onnx", "decoder_model_merged.onnx"), load_external_data=False)
    out = {}
    for t in m.graph.initializer:
        if t.name in ("cos_cache", "sin_cache"):
            loc, off, n = ext(t)
            local = os.path.join(ref, "onnx", loc)
            if os.path.exists(local) and os.path.getsize(local) >= off + n:
                out[t.name] = np.array(read_ext(os.path.join(ref, "onnx"), t))
            else:
                out[t.name] = np.frombuffer(hub_range("onnx/" + loc, off, n), np.float32).reshape(list(t.dims))
    return out


def build(ref, out, variants, graphs, src):
    rope = rope_caches(ref) if "decoder_model_merged" in graphs else {}
    os.makedirs(os.path.join(out, "onnx"), exist_ok=True)
    report = {}
    for graph in graphs:
        names = [t.name for t in onnx.load(os.path.join(ref, "onnx", graph + ".onnx"), load_external_data=False).graph.initializer]
        for variant in variants:
            fname = graph + SUFFIX[variant] + ".onnx"
            t0 = time.time()
            model = onnx.load(os.path.join(ref, "onnx", fname), load_external_data=False)
            jobs = plan(graph, model)
            ends = {}
            for t in model.graph.initializer:
                if t.external_data:
                    loc, off, n = ext(t)
                    ends[loc] = max(ends.get(loc, 0), off + n)
            handles = {}
            for loc, size in ends.items():
                f = open(os.path.join(out, "onnx", loc), "wb")
                f.truncate(size)
                handles[loc] = f

            def put(t, arr):
                loc, off, n = ext(t)
                if arr.dtype != NP[t.data_type] or arr.size != int(np.prod(t.dims)):
                    raise SystemExit(f"{t.name}: got {arr.dtype}{list(arr.shape)}, want {NP[t.data_type]}{list(t.dims)}")
                b = np.ascontiguousarray(arr).tobytes()
                if len(b) != n:
                    raise SystemExit(f"{t.name}: {len(b)} bytes != {n}")
                handles[loc].seek(off)
                handles[loc].write(b)

            for kind, base, parts in jobs:
                if kind == "plain":
                    t = parts["t"]
                    w = rope[t.name] if t.name in rope else src.get(graph, t.name)
                    put(t, w.astype(NP[t.data_type]))
                    continue
                if graph == "embed_tokens":
                    w = src.get(graph, "model.embed_tokens.weight").T
                else:
                    w = src.get(graph, fp32_name(graph, base, names))
                w = np.ascontiguousarray(w.astype(NP[parts["s"].data_type]))
                if kind == "sym4":
                    q, s = sym4(w)
                    put(parts["q"], q)
                    put(parts["s"], s)
                else:
                    qdims = list(parts["q"].dims)
                    per_row = qdims[-1] if len(qdims) == 3 else qdims[-1] * 32 // w.shape[0]
                    q, s, z = asym(w, 4 if per_row == 16 else 8)
                    put(parts["q"], q)
                    put(parts["s"], s)
                    put(parts["z"], z)
            for f in handles.values():
                f.close()
            shutil.copyfile(os.path.join(ref, "onnx", fname), os.path.join(out, "onnx", fname))
            report[fname] = ends
            print(f"{fname}: {len(jobs)} tensors, {sum(ends.values()) / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", required=True, choices=list(SUFFIX))
    ap.add_argument("--graphs", nargs="+", default=GRAPHS, choices=GRAPHS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt")
    g.add_argument("--src-onnx")
    a = ap.parse_args()
    src = Checkpoint(a.ckpt) if a.ckpt else OnnxSource(a.src_onnx)
    report = build(a.ref, a.out, a.variants, a.graphs, src)
    json.dump(report, open(os.path.join(a.out, f"build-{'-'.join(a.variants)}.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
