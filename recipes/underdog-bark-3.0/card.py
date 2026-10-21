"""Fill card/README.md from the parity results and the files in $OUT/onnx."""
import argparse, json, os

SUFFIX = {"fp32": "", "fp16": "_fp16", "q8": "_quantized", "q4": "_q4", "q4f16": "_q4f16"}
ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--variants", nargs="+", required=True)
a = ap.parse_args()
e = os.environ
parity = json.load(open(f"{a.results}/parity.json"))


def size(stem):
    return sum(os.path.getsize(f"{a.out}/onnx/{f}") for f in os.listdir(f"{a.out}/onnx") if f.startswith(stem + "."))


rows = []
for v in a.variants:
    s = SUFFIX[v]
    total = size(f"encoder_model{s}") + size(f"decoder_joint_model{s}")
    rows.append(f"| {v} | `onnx/encoder_model{s}.onnx` | `onnx/decoder_joint_model{s}.onnx` | {total / 1e6:,.0f} MB |")
n = parity["clips"]
prow = [f"| PyTorch fp32 | {n}/{n} | 0 | {parity['pytorch']['wer_vs_script']:.2%} |"]
for v in a.variants:
    r = parity["variants"][v]
    prow.append(f"| {v} | {r['exact_tokens_and_durations']}/{n} | {r['wer_vs_pytorch']:.2%} | {r['wer_vs_script']:.2%} |")
card = (open(a.template).read().replace("__SOURCE_REPO__", e["RECIPE_SOURCE_REPO"])
        .replace("__SOURCE_REVISION__", e["RECIPE_SOURCE_REVISION"][:7]).replace("__VARIANT_ROWS__", "\n".join(rows))
        .replace("__PARITY_ROWS__", "\n".join(prow)).replace("__CLIPS__", str(n)))
open(f"{a.out}/README.md", "w").write(card)
