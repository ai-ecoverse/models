"""Write the non-ONNX files of the export: configs, tokenizer, NOTICE.

preprocessor_config.json keeps transformers' ParakeetFeatureExtractor keys (so AutoFeatureExtractor still loads it)
and adds every parameter a client needs to reproduce the features without transformers.

    python assemble.py --hf hf --src src --out $OUT
"""
import argparse, json, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

for f in ("config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
    shutil.copy(f"{a.hf}/{f}", f"{a.out}/{f}")
shutil.copy(f"{a.src}/NOTICE", f"{a.out}/NOTICE")

fe = json.load(open(f"{a.hf}/processor_config.json"))["feature_extractor"]
pp = {
    **fe,
    # transformers keys above; the rest documents ParakeetFeatureExtractor (transformers 5.20) exactly
    "dither": 0.0,
    "leading_silence_seconds": 0.25,
    "window": "hann",
    "window_periodic": False,
    "window_centered_in_n_fft": True,
    "center": True,
    "pad_mode": "constant",
    "preemphasis_first_sample": "kept",
    "preemphasis_masks_padding": True,
    "power": 2.0,
    "mel_scale": "slaney",
    "mel_norm": "slaney",
    "mel_fmin": 0.0,
    "mel_fmax": fe["sampling_rate"] / 2,
    "log": "natural",
    "log_zero_guard_type": "add",
    "log_zero_guard_value": 2.0 ** -24,
    "normalize": "per_feature",
    "normalize_over": "valid_frames",
    "normalize_variance": "unbiased",
    "normalize_epsilon": 1e-5,
    "normalize_epsilon_added_to": "std",
    "frames": "1 + floor(num_samples / hop_length)",
    "valid_frames": "floor(num_samples / hop_length)",
    "padded_frames_value": 0.0,
}
json.dump(pp, open(f"{a.out}/preprocessor_config.json", "w"), indent=2)
print("assembled")
