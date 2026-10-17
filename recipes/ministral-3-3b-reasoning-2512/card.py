"""Fill the model card template from what is on disk: the variants present in OUT, the parity results and the self-test.

    python card.py --template card/README.md --out OUT --results results --variants q4f16 [fp16 q4 quantized fp32] \
        [--missing fp32="reason"]
"""
import argparse, glob, json, os, sys

SUFFIX = {"fp32": "", "fp16": "_fp16", "q4": "_q4", "q4f16": "_q4f16", "quantized": "_quantized"}
GRAPHS = ["decoder_model_merged", "embed_tokens", "vision_encoder"]
WHAT = {
    "q4f16": "4-bit weights (block 32) with fp16 activations, KV cache and scales; 8-bit embeddings. The default.",
    "q4": "4-bit weights (block 32), fp32 activations and KV cache",
    "fp16": "fp16 weights, activations and KV cache",
    "quantized": "8-bit weights (block 32), fp32 activations and KV cache",
    "fp32": "fp32 weights (the bf16 checkpoint, widened)",
}
PROMPTS = {"think": "bat-and-ball puzzle (reasoning)", "tool": "weather tool call", "terse": "one-word answer, own system prompt",
           "german": "German explanation (reasoning)", "image": "describe a synthetic image (reasoning, vision)"}


def size(out, variant, graph):
    base = os.path.join(out, "onnx", graph + SUFFIX[variant] + ".onnx")
    files = [base] + sorted(glob.glob(base + "_data*"))
    return sum(os.path.getsize(f) for f in files if os.path.exists(f)), len(files) - 1


def final(text):
    if text.startswith("[THINK]") and "[/THINK]" not in text:
        return None
    return " ".join(text.split("[/THINK]")[-1].replace("</s>", "").split())


def gb(n):
    return f"{n / 1e9:.2f} GB"


def variants_table(out, variants, missing):
    rows = ["| Variant | Decoder | Embeddings | Vision encoder | Total | What |", "|---|---|---|---|---|---|"]
    for v in ["q4f16", "q4", "fp16", "quantized", "fp32"]:
        if v in variants:
            s = [size(out, v, g) for g in GRAPHS]
            rows.append(f"| `{v}` | {gb(s[0][0])} ({s[0][1]} data file{'s' if s[0][1] > 1 else ''}) | {gb(s[1][0])} | {gb(s[2][0])} | "
                        f"{gb(sum(x[0] for x in s))} | {WHAT[v]} |")
        elif v in missing:
            rows.append(f"| `{v}` | not uploaded | | | | {missing[v]} |")
    return "\n".join(rows)


def parity_section(results, variants):
    torch = json.load(open(os.path.join(results, "torch", "torch.json")))["torch"]
    lines = ["Greedy decoding of five prompts, tokenized once with this repo's tokenizer and chat template, against "
             "PyTorch (transformers, `Mistral3ForConditionalGeneration`, fp32 on CPU) running the base checkpoint. "
             "ONNX Runtime 1.31 on CPU. *Greedy match* counts the tokens identical to PyTorch's before the first "
             "different one (of PyTorch's total). The teacher-forced columns feed PyTorch's own tokens and compare the "
             "next-token distributions at every position: argmax agreement, mean KL(PyTorch ‖ ONNX) in nats, and the "
             "largest logit difference.", ""]
    lines.append("| Prompt | PyTorch tokens | " + " | ".join(f"`{v}` greedy match" for v in variants) + " |")
    lines.append("|---|---|" + "---|" * len(variants))
    res = {v: json.load(open(os.path.join(results, f"onnx-{v}.json")))["prompts"] for v in variants if os.path.exists(os.path.join(results, f"onnx-{v}.json"))}
    for p, label in PROMPTS.items():
        if p not in torch:
            continue
        cells = []
        for v in variants:
            r = res.get(v, {}).get(p)
            if r is None:
                cells.append("—")
            elif r["identical"]:
                cells.append("identical")
            elif r.get("max_new") and r["max_new"] < torch[p]["new_tokens"] and r["first_divergence"] >= r["max_new"]:
                cells.append(f"first {r['max_new']} identical (capped)")
            else:
                a, b = final(torch[p]["text"]), final(r["text"])
                note = "" if a is None or b is None else ("; same answer" if a == b else "; different answer")
                cells.append(f"{r['first_divergence']}{note}")
        lines.append(f"| {label} | {torch[p]['new_tokens']} | " + " | ".join(cells) + " |")
    lines += ["", "| Variant | teacher-forced argmax agreement | mean KL | max KL | max \\|Δlogit\\| | vision features: max \\|Δ\\| / min cosine |",
              "|---|---|---|---|---|---|"]
    for v in variants:
        if v not in res:
            continue
        rs = list(res[v].values())
        n = sum(r["torch_tokens"] for r in rs)
        agree = sum(r["tf_argmax_agree"] * r["torch_tokens"] for r in rs) / n
        kl = sum(r["tf_mean_kl"] * r["torch_tokens"] for r in rs) / n
        im = res[v].get("image", {})
        vis = f"{im['vision_max_abs']:.2g} / {im['vision_min_cos']:.6f}" if "vision_max_abs" in im else "—"
        lines.append(f"| `{v}` | {agree * 100:.2f}% | {kl:.2g} | {max(r['tf_max_kl'] for r in rs):.2g} | "
                     f"{max(r['tf_max_abs_logit'] for r in rs):.3g} | {vis} |")
    lines += ["", "Greedy decoding of a reasoning model at temperature 0 is a stress test: one flipped near-tie changes "
              "everything after it, so a divergence after many identical tokens is expected for the quantized variants "
              "and says little about quality on its own; the teacher-forced numbers measure the drift directly. "
              "*same answer* means the text after `[/THINK]` (or the tool call) is identical to PyTorch's; the "
              "bat-and-ball run stops at 1024 tokens while still thinking, on both sides."]
    capped = sorted({(v, r["max_new"]) for v in res for p, r in res[v].items()
                     if r.get("max_new") and p in torch and r["max_new"] < torch[p]["new_tokens"]})
    for v, n in capped:
        lines += ["", f"The `{v}` greedy runs stop after {n} tokens per prompt: ONNX Runtime's CPU kernel for "
                  "8-bit `MatMulNBits` decodes slowly, one token at a time. Its teacher-forced numbers still cover "
                  "all of PyTorch's tokens."]
    vis = os.path.join(results, "vision.txt")
    if os.path.exists(vis):
        lines += ["", open(vis).read().strip()]
    return "\n".join(lines)


def selftest_section(path, out):
    st = json.load(open(path))
    same, diff = [], []
    for fname, files in st.items():
        total = sum(f["bytes_differing"] for f in files.values())
        (same if total == 0 else diff).append((fname, files))
    lines = [f"Byte-identical to Mistral's files: {', '.join(f'`{f}`' for f, _ in same)}.", ""]
    for fname, files in diff:
        tensors = sorted({t[0] for f in files.values() for t in f["tensors"]})
        nbytes = sum(f["bytes_differing"] for f in files.values())
        data = sum(os.path.getsize(os.path.join(out, "onnx", loc)) for loc in files)
        where = ("all in the projector's `patch_merger` weight and its scales"
                 if all("patch_merger" in t for t in tensors) else "spread over the 4-bit weights")
        lines.append(f"- `{fname}`: {nbytes:,} of {data:,} bytes differ ({nbytes / data * 100:.2g}%), {where}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--missing", nargs="*", default=[])
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--parity", nargs="*")
    a = ap.parse_args()
    missing = dict(m.split("=", 1) for m in a.missing)
    text = open(a.template).read()
    text = text.replace("__VARIANTS__", variants_table(a.out, a.variants, missing))
    text = text.replace("__PARITY__", parity_section(a.results, a.parity or a.variants))
    prov = open(a.provenance).read().replace("__SELFTEST__", selftest_section(os.path.join(a.results, "selftest.json"), a.out))
    prov = prov.replace("__BUILT_ON__", os.environ.get("MODELS_BUILT_ON", "a local machine"))
    text = text.replace("__PROVENANCE__", prov.strip())
    open(os.path.join(a.out, "README.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
