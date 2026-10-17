"""Fail the build when an exported variant drifts further from PyTorch than the 2026-10-17 export did.

    python gate.py RESULTS q4f16 q4 fp16 quantized fp32

Teacher-forced numbers, token-weighted over all prompts (see parity.py). The 2026-10-17 export measured q4f16/q4
95.1%/95.2% argmax agreement and mean KL 0.021, quantized 99.7% / 8e-5, fp16 and fp32 99.9% / ≤2e-6.
"""
import json, os, sys

FLOOR = {  # argmax agreement ≥, mean KL ≤
    "q4f16": (0.93, 0.04),
    "q4": (0.93, 0.04),
    "quantized": (0.99, 1e-3),
    "fp16": (0.995, 1e-4),
    "fp32": (0.995, 1e-4),
}


def main():
    results, variants = sys.argv[1], sys.argv[2:]
    ok = True
    for v in variants:
        rs = list(json.load(open(os.path.join(results, f"onnx-{v}.json")))["prompts"].values())
        n = sum(r["torch_tokens"] for r in rs)
        agree = sum(r["tf_argmax_agree"] * r["torch_tokens"] for r in rs) / n
        kl = sum(r["tf_mean_kl"] * r["torch_tokens"] for r in rs) / n
        min_agree, max_kl = FLOOR[v]
        good = agree >= min_agree and kl <= max_kl
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {v}: argmax agreement {agree:.4f} (≥ {min_agree}), mean KL {kl:.2g} (≤ {max_kl:g})")
    if "fp32" in variants:
        rs = json.load(open(os.path.join(results, "onnx-fp32.json")))["prompts"]
        same = sum(r["identical"] for r in rs.values())
        print(f"fp32 greedy identical on {same}/{len(rs)} prompts")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
