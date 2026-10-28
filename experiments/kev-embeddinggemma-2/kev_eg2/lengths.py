"""Token lengths of decision-v7 (train, calibration, development) and transfer-v4 development under EmbeddingGemma 2's
tokenizer, as the training rows are built (upstream's augmentation for both epochs, none pairs included): how many rows
fit a cap, and how many records upstream's strict training context (384 state / 1024 row / 2048 packed) rejects.

    python -m kev_eg2.lengths --out runs/lengths.json
"""
import argparse, json, os
from collections import Counter
from types import SimpleNamespace
from . import adapt
from kev.data import materialize
from kev.model import ContextOverflow, encode, rows_of, training_context
from kev.suite import load_split
from kev.train import record_variants
from transformers import AutoTokenizer

CAPS = (256, 384, 512, 640, 768, 1024)


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def measure(tok, reqs, augment_epochs=0, seed=0):
    c = training_context()
    a = SimpleNamespace(seed=seed, p_none=0.1, p_none_distract=0.12, p_distract=0.15, p_none_pair=0.25)
    rows, states, rejected, truncated_states = [], [], Counter(), 0
    for r in reqs:
        variants = [r] if not augment_epochs else [v for ep in range(augment_epochs) for v in record_variants(r, a, ep)[0]]
        for v in variants:
            rec = materialize(v)
            try:
                enc = encode(tok, rec, max_state=c["max_state"], max_branch=c["max_branch"], strict=True)
                if len(enc["ids"]) > c["max_packed"]: rejected["packed"] += 1; continue
            except ContextOverflow as e:
                rejected["state" if e.max_state else "branch"] += 1; continue
            S, _, rs = rows_of(enc)
            states.append(len(S))
            rows += [len(S) + len(x["ids"]) + 2 for x in rs]   # + <bos>, <eos>
    return {"variants": len(states) + sum(rejected.values()), "rows": len(rows), "rejected": dict(rejected),
            "state_p50_p99_max": [pct(states, .5), pct(states, .99), max(states)],
            "row_p50_p90_p99_max": [pct(rows, .5), pct(rows, .9), pct(rows, .99), max(rows)],
            "rows_within": {cap: sum(x <= cap for x in rows) / len(rows) for cap in CAPS},
            "rows_over": {cap: sum(x > cap for x in rows) for cap in CAPS}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); a.out = os.path.abspath(a.out)
    os.chdir(adapt.KEV_ROOT)
    tok = AutoTokenizer.from_pretrained(adapt.BASE, revision=adapt.BASE_REVISION)
    out = {"train (2 epochs, augmented)": measure(tok, load_split("evals/v7/decision-v7", "train"), augment_epochs=2)}
    for suite, split in (("evals/v7/decision-v7", "calibration"), ("evals/v7/decision-v7", "development"), ("evals/v4/transfer-v4", "development")):
        out[f"{suite} {split}"] = measure(tok, load_split(suite, split))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
