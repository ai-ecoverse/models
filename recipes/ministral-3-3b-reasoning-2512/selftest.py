"""Self-test of build.py: rebuild the reference's own variants from the reference's fp32 graphs and compare every
external-data file byte for byte with the reference.

    python selftest.py --ref REF --tmp /tmp/selftest [--variants fp16 q4 q4f16 quantized] [--out results/selftest.json] \
        [--tolerance 0.0005]

--tolerance passes the test when, in every file, at most that fraction of the bytes differs (onnxruntime's 4-bit
quantizer rounds boundary values differently on x86 and ARM).
"""
import argparse, json, os, shutil, sys
import numpy as np
import onnx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import GRAPHS, SUFFIX, OnnxSource, build, ext


def compare(ref, tmp, fname):
    m = onnx.load(os.path.join(ref, "onnx", fname), load_external_data=False)
    files = sorted({ext(t)[0] for t in m.graph.initializer if t.external_data})
    out = {}
    for loc in files:
        a, b = os.path.join(ref, "onnx", loc), os.path.join(tmp, "onnx", loc)
        same_size = os.path.getsize(a) == os.path.getsize(b)
        diff = 0
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                x, y = fa.read(1 << 26), fb.read(1 << 26)
                if not x and not y:
                    break
                n = min(len(x), len(y))
                diff += int((np.frombuffer(x[:n], np.uint8) != np.frombuffer(y[:n], np.uint8)).sum()) + abs(len(x) - len(y))
        bad = []
        if diff:
            for t in m.graph.initializer:
                if t.external_data and ext(t)[0] == loc:
                    _, off, n = ext(t)
                    with open(a, "rb") as fa, open(b, "rb") as fb:
                        fa.seek(off), fb.seek(off)
                        k = int((np.frombuffer(fa.read(n), np.uint8) != np.frombuffer(fb.read(n), np.uint8)).sum())
                    if k:
                        bad.append([t.name, k, n])
        out[loc] = {"size_equal": same_size, "bytes_differing": diff, "tensors": bad[:20]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--variants", nargs="+", default=["fp16", "q4", "q4f16", "quantized"])
    ap.add_argument("--out", default="selftest.json")
    ap.add_argument("--tolerance", type=float, default=0.0)
    a = ap.parse_args()
    src = OnnxSource(a.ref)
    res, ok = {}, True
    for v in a.variants:
        for g in GRAPHS:
            shutil.rmtree(a.tmp, ignore_errors=True)
            build(a.ref, a.tmp, [v], [g], src)
            fname = g + SUFFIX[v] + ".onnx"
            r = compare(a.ref, a.tmp, fname)
            res[fname] = r
            same = all(x["size_equal"] and x["bytes_differing"] == 0 for x in r.values())
            good = all(x["size_equal"] and x["bytes_differing"] <= a.tolerance * os.path.getsize(os.path.join(a.ref, "onnx", loc))
                       for loc, x in r.items())
            ok &= good
            print(fname, "IDENTICAL" if same else ("within tolerance" if good else "DIFFERS"),
                  "" if same else {loc: x["bytes_differing"] for loc, x in r.items()}, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    shutil.rmtree(a.tmp, ignore_errors=True)
    print("selftest", "passed" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
