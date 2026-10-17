"""Fail the build when an exported variant drifts too far from PyTorch fp32.

    python gate.py RESULTS q4f16 q4

Teacher-forced numbers, token-weighted over all prompts (parity.py), plus the fp16 vision encoder against
PyTorch's image features. Mistral's own 3B q4f16/q4 graphs measure about 95% argmax agreement and mean KL 0.021
on the same prompts (recipes/ministral-3-3b-reasoning-2512); the floors leave room for model-to-model variation
without letting a broken export through.
"""
import json, os, sys

FLOOR = {  # argmax agreement ≥, mean KL ≤
    "q4f16": (0.90, 0.06),
    "q4": (0.90, 0.06),
    "fp16": (0.995, 1e-4),
    "fp32": (0.995, 1e-4),
}
VISION_MIN_COS = 0.99


def main():
    results, variants = sys.argv[1], sys.argv[2:]
    ok = True
    for v in variants:
        rs = json.load(open(os.path.join(results, f"onnx-{v}.json")))["prompts"]
        n = sum(r["torch_tokens"] for r in rs.values())
        agree = sum(r["tf_argmax_agree"] * r["torch_tokens"] for r in rs.values()) / n
        kl = sum(r["tf_mean_kl"] * r["torch_tokens"] for r in rs.values()) / n
        min_agree, max_kl = FLOOR[v]
        good = agree >= min_agree and kl <= max_kl
        print(f"{'ok ' if good else 'BAD'} {v}: argmax agreement {agree:.4f} (≥ {min_agree}), mean KL {kl:.2g} (≤ {max_kl:g})")
        ok &= good
        im = rs.get("image", {})
        if "vision_min_cos" in im:
            vgood = im["vision_min_cos"] >= VISION_MIN_COS
            print(f"{'ok ' if vgood else 'BAD'} {v}: fp16 vision features min cosine {im['vision_min_cos']:.6f} (≥ {VISION_MIN_COS}), max |Δ| {im['vision_max_abs']:.3g}")
            ok &= vgood
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
