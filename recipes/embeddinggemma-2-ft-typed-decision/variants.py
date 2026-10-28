"""Build the encoder's other variants from the fp32 export, in onnx-community's own graphs.

Each variant reuses onnx-community/embeddinggemma-2-ONNX's graph for it unchanged and writes its external data at the
reference's offsets, from the fp32 weights of `--src` (this recipe's fp32 export), quantized the reference's way:

  *_quant + *_scales + *_zp   asymmetric 4/8-bit blocks of 32 along K (MatMulNBits) or along the hidden axis of the
                              token embedding (GatherBlockQuantized), onnxruntime's quant_slice_asymmetric with np.round,
                              computed in fp32; the f16 variants quantize the fp16-rounded weights and store fp16
                              scales
  plain                       cast to the reference tensor's dtype (fp32 or fp16)

  q8     model_quantized   int8 weights, fp32 activations (kev.js calls this q8f32)
  fp16   model_fp16        fp16 weights and activations
  q4     model_q4          4-bit weights, fp32 activations
  q4f16  model_q4f16       4-bit weights, fp16 activations
  q8f16  model_q8f16       int8 weights, fp16 activations: not published by onnx-community, so q4f16's graph with
                           8-bit blocks (bits=8 and 8-bit shapes), laid out anew

    python variants.py --ref REF --src RAW --out OUT/onnx --variants q8 fp16 q4 q4f16 q8f16
    python variants.py --ref REF --src REF --check --variants q8 fp16 q4 q4f16 --report selftest.json

--check builds every variant from the reference's own fp32 graph and compares each tensor with the reference's
bytes, so the quantization is the one onnx-community used.
"""
import argparse, json, os, sys, time
import numpy as np
import onnx
from onnx import helper, numpy_helper

SUFFIX = {"fp32": "", "q8": "_quantized", "fp16": "_fp16", "q4": "_q4", "q4f16": "_q4f16", "q8f16": "_q8f16"}
REF_GRAPH = {"q8f16": "_q4f16"}   # variants built on another variant's graph
NP = {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16, onnx.TensorProto.UINT8: np.uint8}
BLOCK = 32


def ext(t):
    d = {e.key: e.value for e in t.external_data}
    return d["location"], int(d.get("offset", 0)), int(d["length"])


def read_ext(base, t):
    loc, off, n = ext(t)
    with open(os.path.join(base, loc), "rb") as f:
        f.seek(off)
        b = f.read(n)
    if len(b) != n:
        raise SystemExit(f"{t.name}: short read from {loc}")
    return np.frombuffer(b, NP[t.data_type]).reshape(list(t.dims))


class Fp32:
    """fp32 weights by initializer name, from an fp32 model.onnx (external data next to it)."""

    def __init__(self, path):
        self.dir = os.path.dirname(path)
        self.inits = {t.name: t for t in onnx.load(path, load_external_data=False).graph.initializer}
        self.by_flat = {n.replace(".", "_"): n for n in self.inits}

    def get(self, name):
        t = self.inits[name if name in self.inits else self.by_flat[name]]
        if t.data_location == onnx.TensorProto.EXTERNAL:
            return np.array(read_ext(self.dir, t), np.float32)
        return numpy_helper.to_array(t).astype(np.float32)


def asym(w, bits, rows=8192):
    """onnxruntime's quant_slice_asymmetric over blocks of 32 along K, in w's dtype, for w of shape (K, N).

    Same routine as recipes/ministral-3-3b-reasoning-2512/build.py, which reproduces Mistral's builder bytes."""
    k, n = w.shape
    if k % BLOCK:
        raise SystemExit("K must be a multiple of 32")
    levels = (1 << bits) - 1
    mid = (levels + 1) // 2
    dt = w.dtype.type
    kb = k // BLOCK
    q = np.empty((n, kb, BLOCK), np.uint8)
    s = np.empty((n, kb), w.dtype)
    z = np.empty((n, kb), np.uint8)
    wt = w.T
    for r0 in range(0, n, rows):
        x = np.ascontiguousarray(wt[r0:r0 + rows]).reshape(-1, kb, BLOCK)
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
    return packed.reshape(n, kb, BLOCK // 2), s, zpk


def gather_nodes(model):
    return {n.input[0][: -len("_quant")] for n in model.graph.node if n.op_type == "GatherBlockQuantized"}


def to_8bit(model):
    """q4f16's graph → q8f16: 8-bit MatMulNBits / GatherBlockQuantized and 8-bit tensor shapes."""
    for n in model.graph.node:
        if n.op_type in ("MatMulNBits", "GatherBlockQuantized"):
            for a in n.attribute:
                if a.name == "bits":
                    a.i = 8
    gathers = gather_nodes(model)
    for t in model.graph.initializer:
        if t.name.endswith("_quant"):
            t.dims[-1] *= 2              # [N, kb, 16] -> [N, kb, 32]; embedding [V, 256] -> [V, 512]
        elif t.name.endswith("_zp"):
            base = t.name[: -len("_zp")]
            scales = next(s for s in model.graph.initializer if s.name == base + "_scales")
            t.dims[-1] = scales.dims[-1]  # one zero point per block, unpacked
    return model


def tensors(model, src, gathers):
    """name -> array for every external initializer of `model`, computed from fp32 `src`."""
    inits = {t.name: t for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL}
    out = {}
    for name, t in inits.items():
        if name.endswith(("_scales", "_zp")):
            continue
        if name.endswith("_quant"):
            base = name[: -len("_quant")]
            s, z = inits[base + "_scales"], inits[base + "_zp"]
            w = src.get(base)
            if base in gathers:
                w = w.T                    # [V, H] embedding: blocks along H, so quantize it as a (H, V) matrix
            if s.data_type == onnx.TensorProto.FLOAT16:
                w = w.astype(np.float16).astype(np.float32)   # f16 variants quantize the fp16-rounded weights
            per_row = t.dims[-1] if len(t.dims) == 3 else t.dims[-1] * BLOCK // w.shape[0]
            q, sc, zp = asym(np.ascontiguousarray(w), 4 if per_row == BLOCK // 2 else 8)
            out[name], out[base + "_scales"], out[base + "_zp"] = q, sc.astype(NP[s.data_type]), zp
        else:
            out[name] = src.get(name).astype(NP[t.data_type])
    return out


def write(model, arrays, out_dir, fname, keep_offsets):
    inits = [t for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    loc = fname + "_data"
    if not keep_offsets:   # lay the data out anew, 64-byte aligned
        off = 0
        for t in inits:
            n = arrays[t.name].nbytes
            del t.external_data[:]
            for k, v in (("location", loc), ("offset", str(off)), ("length", str(n))):
                t.external_data.add(key=k, value=v)
            off += (n + 63) // 64 * 64
    ends = {}
    for t in inits:
        l, o, n = ext(t)
        ends[l] = max(ends.get(l, 0), o + n)
    handles = {}
    for l, size in ends.items():
        f = open(os.path.join(out_dir, l), "wb")
        f.truncate(size)
        handles[l] = f
    for t in inits:
        l, o, n = ext(t)
        a = np.ascontiguousarray(arrays[t.name])
        if a.dtype != NP[t.data_type] or a.size != int(np.prod(t.dims)) or a.nbytes != n:
            raise SystemExit(f"{t.name}: built {a.dtype}{list(a.shape)}, graph wants {NP[t.data_type]}{list(t.dims)}")
        handles[l].seek(o)
        handles[l].write(a.tobytes())
    for f in handles.values():
        f.close()
    return ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="onnx-community/embeddinggemma-2-ONNX snapshot (graphs; data for --check)")
    ap.add_argument("--src", required=True, help="folder with the fp32 onnx/model.onnx to quantize")
    ap.add_argument("--out", help="output onnx/ folder")
    ap.add_argument("--variants", nargs="+", required=True, choices=[v for v in SUFFIX if v != "fp32"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    src = Fp32(os.path.join(a.src, "onnx", "model.onnx"))
    report, bad = {}, []
    for v in a.variants:
        t0 = time.time()
        graph_file = os.path.join(a.ref, "onnx", f"model{REF_GRAPH.get(v, SUFFIX[v])}.onnx")
        model = onnx.load(graph_file, load_external_data=False)
        gathers = gather_nodes(model)
        if v in REF_GRAPH:
            if a.check:
                continue           # nothing to compare with
            model = to_8bit(model)
        arrays = tensors(model, src, gathers)
        if a.check:
            diff = []
            for t in model.graph.initializer:
                if t.data_location == onnx.TensorProto.EXTERNAL:
                    r, b = read_ext(os.path.join(a.ref, "onnx"), t), np.ascontiguousarray(arrays[t.name])
                    if r.tobytes() != b.tobytes():
                        frac = float((r.reshape(-1) != b.reshape(-1)).mean()) if r.size == b.size else 1.0
                        diff.append({"name": t.name, "fraction_differing": frac})
            report[v] = {"tensors": len(arrays), "differing": diff}
            print(f"selftest {v}: {len(arrays)} tensors, {len(diff)} differ from the reference", flush=True)
            if diff:
                bad.append(v)
            continue
        os.makedirs(a.out, exist_ok=True)
        fname = f"model{SUFFIX[v]}.onnx"
        ends = write(model, arrays, a.out, fname, keep_offsets=v not in REF_GRAPH)
        if v in REF_GRAPH:
            onnx.save_model(model, os.path.join(a.out, fname))
        else:
            with open(graph_file, "rb") as f, open(os.path.join(a.out, fname), "wb") as g:
                g.write(f.read())   # the reference graph, unchanged
        report[v] = {"files": ends, "seconds": round(time.time() - t0, 1)}
        print(f"built {v}: {sum(ends.values()) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    if bad:
        raise SystemExit(f"selftest: {', '.join(bad)} differ from the reference")


if __name__ == "__main__":
    sys.exit(main())
