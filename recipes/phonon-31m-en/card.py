import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity import EOS_TIE

LABELS = {
    "fp32": ("", "float32"),
    "q8": ("_quantized", "8-bit `MatMulNBits` (block 32, symmetric, accuracy level 4) on the flow LM; Mimi in fp32"),
    "q4": ("_q4", "4-bit `MatMulNBits` (block 32, symmetric, accuracy level 4) on the flow LM; Mimi in fp32"),
    "fp16": ("_fp16", "float16 weights and activations, float32 inputs and outputs; RoPE's Sin, Cos and Range in fp32"),
}


def size(out, files):
    return sum(os.path.getsize(os.path.join(out, f)) for f in files.values())


def row(v, r):
    return (
        f"| `{v}` | {r['latent_max']:.1e} | {r['latent_rel_p50']:.1e} / {r['latent_rel_p99']:.1e} | "
        f"{r['eos_max']:.1e} | {r['mimi_snr_min']:.1f} / {r['mimi_snr_median']:.1f} dB |"
    )


def free_text(free):
    text = (
        f"stops on the same frame as ptts in {free['same_length']} of {free['sentences']} sentences, "
        f"PCM SNR {free['pcm_snr_min']:.1f} dB min and {free['pcm_snr_median']:.1f} dB median"
    )
    excused = []
    if free["near_ties"]:
        excused.append(
            f"{free['near_ties']} that stops on another frame on an end-of-speech near-tie "
            f"(ptts's logit within {EOS_TIE:g} of the threshold)"
        )
    if free["ill_conditioned"]:
        excused.append(
            f"{free['ill_conditioned']} whose ptts trajectory a float64 PyTorch run with the same noise misses too"
        )
    return text + (f", not counting {' and '.join(excused)}" if excused else "")


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
        rows.append(row(v, tf[v]))
    free = parity["free_running"]
    deltas = sorted(parity["duration_delta"].items(), key=lambda x: float(x[0]))
    delta_rows = [f"| {d} {row(v, run['teacher_forced'][v])}" for d, run in deltas for v in LABELS]
    delta_free = "; ".join(f"at {d}, {free_text(run['free_running'])}" for d, run in deltas)
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
        "__FREE__": free_text(free) + " (f32 rounding compounds over the autoregression)",
        "__DELTAS__": " and ".join(d for d, _ in deltas),
        "__DELTA_PARITY__": "\n".join(delta_rows),
        "__DELTA_FREE__": delta_free,
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
