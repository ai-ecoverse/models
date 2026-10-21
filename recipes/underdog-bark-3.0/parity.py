"""Parity gate and engine fixtures.

Reference: transformers' ParakeetFeatureExtractor + ParakeetForTDT.generate() on the unpacked fp32 weights.
For each clip in clips/ (macOS `say`, 16 kHz) and each variant: the export's encoder_model + decoder_joint_model
through tdt.py (NumPy mel, greedy TDT), compared with the reference.

Gates: tdt.py's mel within 5e-4 of transformers' on every clip; fp32 and fp16 give the reference's exact tokens and
durations on every clip; quantized variants stay within a word error rate of the reference (case and punctuation
folded) and are no worse against the scripts than the reference by more than 2 points.

Writes $RESULTS/parity.json and $OUT/fixture/ (s01: audio, transformers' mel, mel filters, encoder output, and the
expected tokens/durations/text per clip and variant) for the slicc-ortllama parakeet engine's tests.

    python parity.py --hf hf --out $OUT --clips clips --results $RESULTS --variants fp32 fp16 q8 q4f16
"""
import argparse, glob, json, os, re, shutil, sys, time
import numpy as np, onnxruntime as ort, soundfile as sf, torch, jiwer
from transformers import AutoProcessor, ParakeetForTDT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdt  # noqa: E402

SUFFIX = {"fp32": "", "fp16": "_fp16", "q8": "_quantized", "q4": "_q4", "q4f16": "_q4f16"}
MAX_WER = {"q8": 0.03, "q4": 0.10, "q4f16": 0.10}   # vs the reference, normalized
EXACT = {"fp32", "fp16"}
MEL_TOL = 5e-4

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--clips", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--variants", nargs="+", required=True)
a = ap.parse_args()


def norm(t):
    t = t.lower().replace("%", " percent")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", t)).strip()


proc = AutoProcessor.from_pretrained(a.hf)
pp = json.load(open(f"{a.out}/preprocessor_config.json"))
filters = tdt.mel_filters(pp)
model = ParakeetForTDT.from_pretrained(a.hf).eval()
cfg = model.config
gen = dict(vocab_size=cfg.vocab_size, blank_token_id=cfg.blank_token_id, durations=list(cfg.durations),
           decoder_hidden_size=cfg.decoder_hidden_size, num_decoder_layers=cfg.num_decoder_layers,
           max_symbols_per_step=cfg.max_symbols_per_step, eos_token_id=model.generation_config.eos_token_id)
pieces = proc.tokenizer.convert_ids_to_tokens(list(range(cfg.vocab_size)))
problems = []

clips = sorted(glob.glob(f"{a.clips}/*.flac"))
ref = {}
for f in clips:
    name = os.path.basename(f)[:-5]
    w, sr = sf.read(f, dtype="float32", always_2d=True)
    assert sr == pp["sampling_rate"], (f, sr)
    w = tdt.lead(w.mean(1), pp)
    inp = proc([w], sampling_rate=sr, return_tensors="pt")
    feats, mask = tdt.mel(w, pp, filters)
    mel_d = float(np.abs(feats - inp["input_features"].numpy()).max())
    if mel_d > MEL_TOL or not (mask == inp["attention_mask"].numpy()).all():
        problems.append(f"{name}: tdt.mel differs from transformers by {mel_d:.2e}")
    with torch.inference_mode():
        o = model.generate(**inp)
    seq, dur = o.sequences[0].tolist()[1:], o.durations[0].tolist()[1:]
    ref[name] = dict(wave=w, hf_feats=inp["input_features"].numpy(), feats=feats, mask=mask, tokens=seq,
                     durations=dur, mel_max_abs_diff=mel_d,
                     text=proc.tokenizer.decode(seq, skip_special_tokens=True, group_tokens=False),
                     script=open(f[:-5] + ".txt").read().strip())
    assert tdt.detokenize(seq, pieces, cfg.blank_token_id) == ref[name]["text"], name
    print(f"{name}: mel |d| {mel_d:.1e}  {ref[name]['text']}", flush=True)

names = list(ref)
scripts = [norm(ref[n]["script"]) for n in names]
ref_texts = [norm(ref[n]["text"]) for n in names]
ref_vs_script = jiwer.wer(scripts, ref_texts)
report = {"clips": len(names), "mel_max_abs_diff": max(ref[n]["mel_max_abs_diff"] for n in names),
          "pytorch": {"wer_vs_script": round(ref_vs_script, 4)}, "variants": {}}
expected = {n: {"script": ref[n]["script"], "pytorch": {"text": ref[n]["text"], "tokens": ref[n]["tokens"],
                                                         "durations": ref[n]["durations"]}} for n in names}

so = ort.SessionOptions()
so.log_severity_level = 3
fixture = f"{a.out}/fixture"
os.makedirs(fixture, exist_ok=True)
for v in a.variants:
    enc = ort.InferenceSession(f"{a.out}/onnx/encoder_model{SUFFIX[v]}.onnx", so, providers=["CPUExecutionProvider"])
    dj = ort.InferenceSession(f"{a.out}/onnx/decoder_joint_model{SUFFIX[v]}.onnx", so,
                              providers=["CPUExecutionProvider"])
    hyps, exact, t0 = [], 0, time.time()
    for n in names:
        r = ref[n]
        e, el = enc.run(None, {"input_features": r["feats"], "attention_mask": r["mask"]})
        toks, ds = tdt.greedy_tdt(dj, e, int(el[0]), gen)
        hyp = tdt.detokenize(toks, pieces, cfg.blank_token_id)
        exact += toks == r["tokens"] and ds == r["durations"]
        hyps.append(hyp)
        expected[n][v] = {"text": hyp, "tokens": toks, "durations": ds}
        if v == "fp32" and n == names[0]:
            np.save(f"{fixture}/encoder_outputs.npy", e)
    wer_ref = jiwer.wer(ref_texts, [norm(h) for h in hyps])
    wer_script = jiwer.wer(scripts, [norm(h) for h in hyps])
    exact_raw = sum(h == ref[n]["text"] for h, n in zip(hyps, names))
    report["variants"][v] = {"exact_tokens_and_durations": exact, "exact_text": exact_raw,
                             "wer_vs_pytorch": round(wer_ref, 4), "wer_vs_script": round(wer_script, 4),
                             "seconds": round(time.time() - t0, 2)}
    print(v, report["variants"][v], flush=True)
    if v in EXACT and exact != len(names):
        problems.append(f"{v}: exact tokens and durations on {exact}/{len(names)} clips")
    if v in MAX_WER and wer_ref > MAX_WER[v]:
        problems.append(f"{v}: WER vs pytorch {wer_ref:.4f} > {MAX_WER[v]}")
    if wer_script > ref_vs_script + 0.02:
        problems.append(f"{v}: WER vs scripts {wer_script:.4f}, pytorch {ref_vs_script:.4f}")

first = names[0]
shutil.copy(f"{a.clips}/{first}.flac", f"{fixture}/audio.flac")
np.save(f"{fixture}/input_features.npy", ref[first]["hf_feats"])     # transformers' own features
np.save(f"{fixture}/attention_mask.npy", ref[first]["mask"])
np.save(f"{fixture}/mel_filters.npy", filters)
json.dump({"clip": first, "leading_silence_seconds": pp["leading_silence_seconds"],
           "input_features": list(ref[first]["hf_feats"].shape), "encoder_outputs": "fp32 variant",
           "gen": gen, "clips": expected}, open(f"{fixture}/expected.json", "w"), indent=1)

report["problems"] = problems
json.dump(report, open(f"{a.results}/parity.json", "w"), indent=2)
if problems:
    sys.exit("parity gate failed:\n  " + "\n  ".join(problems))
print("parity ok")
