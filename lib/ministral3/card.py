"""Write the model card (README.md) of a Ministral 3 export from the recipe, the files on disk and the parity results.

    python card.py --out OUT --results RESULTS --recipe RECIPE_DIR --variants q4f16 q4

RECIPE_DIR/recipe.yaml gives the names and pins; RECIPE_DIR/about.md is the model-specific paragraph (what the model
is and how it is prompted). Everything else is the same for every Ministral 3 export: card.md in this directory.
"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = {"fp32": "", "fp16": "_fp16", "q4": "_q4", "q4f16": "_q4f16"}
WHAT = {
    "q4f16": "4-bit weights (block 32, symmetric) with fp16 activations and KV cache. The default.",
    "q4": "4-bit weights (block 32, symmetric), fp32 activations and KV cache",
    "fp16": "fp16 weights, activations and KV cache",
    "fp32": "fp32 weights",
}
PROMPTS = {"think": "bat-and-ball puzzle", "tool": "weather tool call", "terse": "one-word answer, own system prompt",
           "german": "German explanation", "image": "describe a synthetic image (vision)"}


def recipe_fields(recipe_dir):
    root = os.path.dirname(os.path.dirname(os.path.abspath(recipe_dir)))
    name = os.path.basename(os.path.abspath(recipe_dir))
    out = subprocess.run(["node", os.path.join(root, "scripts", "read-recipe.mjs"), name, "--json"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def size(out, name):
    base = os.path.join(out, "onnx", name)
    return sum(os.path.getsize(f) for f in [base] + glob.glob(base + "_data*"))


def gb(n):
    return f"{n / 1e9:.2f} GB"


def final(text):
    if text.startswith("[THINK]") and "[/THINK]" not in text:
        return None
    return " ".join(text.split("[/THINK]")[-1].replace("</s>", "").split())


def variants_table(out, variants):
    rows = ["| Variant | Decoder | What |", "|---|---|---|"]
    for v in variants:
        rows.append(f"| `{v}` | {gb(size(out, f'decoder_model_merged{SUFFIX[v]}.onnx'))} | {WHAT[v]} |")
    rows += ["", f"Shared by every variant: `embed_tokens_fp16` ({gb(size(out, 'embed_tokens_fp16.onnx'))}) and "
             f"`vision_encoder_fp16` ({gb(size(out, 'vision_encoder_fp16.onnx'))}). `config.json`'s "
             "`transformers.js_config` names them as the embeddings and vision encoder for every decoder."]
    return "\n".join(rows)


def parity_section(results, variants):
    torch = json.load(open(os.path.join(results, "torch", "torch.json")))["torch"]
    res = {v: json.load(open(os.path.join(results, f"onnx-{v}.json")))["prompts"] for v in variants}
    lines = ["Greedy decoding of five prompts, tokenized once with this repo's tokenizer and chat template, against "
             "PyTorch (transformers, `Mistral3ForConditionalGeneration`, fp32 on CPU) running the base checkpoint, "
             "with ONNX Runtime on CPU. *Greedy match* counts the tokens identical to PyTorch's before the first "
             "different one. The teacher-forced columns feed PyTorch's own tokens and compare the next-token "
             "distributions at every position: argmax agreement, mean KL(PyTorch ‖ ONNX) in nats, and the largest "
             "logit difference. The image prompt runs through `vision_encoder_fp16`.", ""]
    lines.append("| Prompt | PyTorch tokens | " + " | ".join(f"`{v}` greedy match" for v in variants) + " |")
    lines.append("|---|---|" + "---|" * len(variants))
    for p, label in PROMPTS.items():
        if p not in torch:
            continue
        cells = []
        for v in variants:
            r = res[v].get(p)
            if r is None:
                cells.append("—")
            elif r["identical"]:
                cells.append("identical")
            else:
                a, b = final(torch[p]["text"]), final(r["text"])
                note = "" if a is None or b is None else ("; same answer" if a == b else "; different answer")
                cells.append(f"{r['first_divergence']}{note}")
        lines.append(f"| {label} | {torch[p]['new_tokens']} | " + " | ".join(cells) + " |")
    lines += ["", "| Variant | teacher-forced argmax agreement | mean KL | max KL | max \\|Δlogit\\| | `vision_encoder_fp16` vs PyTorch: max \\|Δ\\| / min cosine |",
              "|---|---|---|---|---|---|"]
    for v in variants:
        rs = list(res[v].values())
        n = sum(r["torch_tokens"] for r in rs)
        agree = sum(r["tf_argmax_agree"] * r["torch_tokens"] for r in rs) / n
        kl = sum(r["tf_mean_kl"] * r["torch_tokens"] for r in rs) / n
        im = res[v].get("image", {})
        vis = f"{im['vision_max_abs']:.2g} / {im['vision_min_cos']:.6f}" if "vision_max_abs" in im else "—"
        lines.append(f"| `{v}` | {agree * 100:.2f}% | {kl:.2g} | {max(r['tf_max_kl'] for r in rs):.2g} | "
                     f"{max(r['tf_max_abs_logit'] for r in rs):.3g} | {vis} |")
    lines += ["", "Greedy decoding at temperature 0 is a stress test: one flipped near-tie changes everything after it, "
              "so a divergence after many identical tokens is expected for 4-bit weights and says little about "
              "quality on its own; the teacher-forced numbers measure the drift directly. *same answer* means the "
              "text after `[/THINK]` (or the tool call) is identical to PyTorch's."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    a = ap.parse_args()
    r = recipe_fields(a.recipe)
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    tc = cfg["text_config"]
    import onnxruntime, onnxruntime_genai, transformers, torch
    subs = {
        "__TITLE__": r["card"]["title"],
        "__BASE__": r["source"]["repo"],
        "__BASE_REVISION__": r["source"]["revision"],
        "__REFERENCE__": r["reference"]["repo"],
        "__REFERENCE_REVISION__": r["reference"]["revision"],
        "__REPO__": r["target"]["repo"],
        "__ABOUT__": open(os.path.join(a.recipe, "about.md")).read().strip(),
        "__LAYERS__": str(tc["num_hidden_layers"]),
        "__HIDDEN__": str(tc["hidden_size"]),
        "__VARIANTS__": variants_table(a.out, a.variants),
        "__PARITY__": parity_section(a.results, a.variants),
        "__TOOLS__": (f"onnxruntime-genai {onnxruntime_genai.__version__} (model builder), onnxruntime {onnxruntime.__version__}, "
                      f"transformers {transformers.__version__}, torch {torch.__version__} (CPU)"),
        "__BUILT_ON__": os.environ.get("MODELS_BUILT_ON", "a local machine"),
        "__RECIPE__": os.path.basename(os.path.abspath(a.recipe)),
    }
    text = open(os.path.join(HERE, "card.md")).read()
    for k, v in subs.items():
        text = text.replace(k, v)
    left = sorted(set(re.findall(r"__[A-Z_]+__", text)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(os.path.join(a.out, "README.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
