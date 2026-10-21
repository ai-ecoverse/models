import argparse
import json
import os
import re

LABELS = {
    "fp32": ("", "float32"),
    "q8": ("_quantized", "8-bit `MatMulNBits` (block 32, symmetric, accuracy level 4) on the flow LM; Mimi in fp32"),
    "q4": ("_q4", "4-bit `MatMulNBits` (block 32, symmetric, accuracy level 4) on the flow LM; Mimi in fp32"),
    "fp16": ("_fp16", "float16 weights and activations, float32 inputs and outputs; RoPE's Sin, Cos and Range in fp32"),
}


def size(out, files):
    return sum(os.path.getsize(os.path.join(out, f)) for f in files.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e = os.environ
    parity = json.load(open(os.path.join(a.results, "parity.json")))
    tokens = json.load(open(os.path.join(a.results, "tokens.json")))
    cfg = json.load(open(os.path.join(a.out, "config.json")))
    voices = cfg["onnx"]["voices"]
    tf = parity["teacher_forced"]
    variants = []
    rows = []
    for v, (suffix, what) in LABELS.items():
        files = parity["graphs"][v]
        variants.append(f"| `{v}` | {f'`{suffix}`' if suffix else 'none'} | {what} | {size(a.out, files) / 1e6:.0f} MB |")
        r = tf[v]
        rows.append(
            f"| `{v}` | {r['latent_max']:.1e} | {r['latent_rel_p50']:.1e} / {r['latent_rel_p99']:.1e} | "
            f"{r['eos_max']:.1e} | {r['mimi_snr_min']:.1f} / {r['mimi_snr_median']:.1f} dB |"
        )
    free = parity["free_running"]
    free_text = (
        f"stops on the same frame as ptts in {free['same_length']} of {free['sentences']} sentences, "
        f"PCM SNR {free['pcm_snr_min']:.1f} dB min and {free['pcm_snr_median']:.1f} dB median "
        "(f32 rounding compounds over the autoregression)"
    )
    sets = tokens["sets"]
    token_text = " and ".join(
        f"{s['identical']} of {s['inputs']} {'fuzzed strings' if name.startswith('fuzz') else 'test inputs'}"
        for name, s in sets.items()
    )
    voice_max = max(parity["voices"].values())
    replace = {
        "__SOURCE_REPO__": e["RECIPE_SOURCE_REPO"],
        "__SOURCE_REVISION__": e["RECIPE_SOURCE_REVISION"],
        "__VOICES_REPO__": e["RECIPE_VOICES_REPO"],
        "__VOICES_REVISION__": e["RECIPE_VOICES_REVISION"],
        "__PTTS_VERSION__": e["RECIPE_PTTS_VERSION"],
        "__PTTS_REVISION__": e["RECIPE_PTTS_REVISION"],
        "__DEFAULT_VOICE__": cfg["onnx"]["default_voice"],
        "__MIMI_STATE__": str(len(cfg["onnx"]["mimi_state"])),
        "__VOICE_COUNT__": str(len(voices)),
        "__VOICES__": ", ".join(f"`{v['name']}`" for v in voices),
        "__VARIANTS__": "\n".join(variants),
        "__PARITY__": "\n".join(rows),
        "__SENTENCES__": str(free["sentences"]),
        "__FRAMES__": str(tf["fp32"]["frames"]),
        "__FREE__": free_text,
        "__VOICE_MATCH__": f"within {voice_max:.0e}" if voice_max else "bit for bit",
        "__ORTLLAMA__": tokens["ortllama"],
        "__TOKENS__": token_text,
    }
    text = open(a.template).read()
    for k, v in replace.items():
        text = text.replace(k, v)
    left = sorted(set(re.findall(r"__[A-Z0-9_]+__", text)))
    if left:
        raise SystemExit(f"card placeholders left: {left}")
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
