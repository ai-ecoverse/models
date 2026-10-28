"""Golden fixtures for other runtimes: request -> rendered texts -> token ids -> embeddings -> logits -> probabilities.

    python fixtures.py --src SRC --requests fixtures/requests.json --out OUT/fixtures/typed-decision.json \
        --export OUT --variants fp32 q8 [--onnx-dir DIR --experiments fp16 q4] --results RESULTS/fixtures.json

Every stage comes from the source model at the pinned revision (PyTorch fp32, CPU, `trust_remote_code`), so a port
can be checked stage by stage: the texts byte for byte, the token ids exactly, then the embeddings (cosine), the
logits and the probabilities. Requests the source refuses carry `error` (the exception type) instead. Each variant's
own deviation from these fixtures goes to --results (embedding cosine, max |Δp|), as the bound a port of that
variant can expect.
"""
import argparse, json, os, sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402


def f32(a):
    return [float(x) for x in np.asarray(a, np.float32).reshape(-1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--variants", nargs="+", default=[])
    ap.add_argument("--onnx-dir", help="folder of the experimental variants' graphs")
    ap.add_argument("--experiments", nargs="*", default=[])
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.src, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.src, trust_remote_code=True, dtype=torch.float32).eval()
    out = []
    for fx in json.load(open(a.requests)):
        req, wf = fx["request"], fx.get("workflow", "")
        entry = {"name": fx["name"], "workflow": wf, "request": req}
        try:
            inputs, specs, texts = model._build_inputs([(wf, req["state"], req["questions"], None)], tok)
        except Exception as e:   # noqa: BLE001
            entry["error"] = type(e).__name__
            out.append(entry)
            continue
        with torch.no_grad():
            o = model(**inputs)
        answers = model.decide(req["state"], req["questions"], workflow=wf, tokenizer=tok)
        n = [len(s.options) for s in specs]
        mask = inputs["option_mask"]
        raw = o.logits.masked_fill(~mask, -1e4).softmax(-1)
        ids = [row[: int(m.sum())].tolist() for row, m in zip(inputs["input_ids"], inputs["attention_mask"])]
        entry.update({
            "questions": [{"qid": s.qid, "kind": s.kind, "instructions": s.instructions, "options": list(s.options),
                           "descriptions": list(s.descriptions)} for s in specs],
            "texts": texts,
            "token_ids": ids,
            "layout": {k: inputs[k].tolist() for k in ("state_index", "question_index", "option_index", "option_mask",
                                                       "question_type")},
            "embeddings": [f32(e) for e in o.text_embeds],
            "logits": [f32(row[:k]) for row, k in zip(o.logits, n)],
            "raw_probabilities": [f32(row[:k]) for row, k in zip(raw, n)],
            "probabilities": [f32(row[:k]) for row, k in zip(o.probabilities, n)],
            "answers": {q: v.to_dict() for q, v in answers.items()},
        })
        mine, lay = render.build(req, wf)
        if mine != texts or lay["option_index"] != entry["layout"]["option_index"]:
            raise SystemExit(f"{fx['name']}: render.py differs from the source")
        out.append(entry)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"format": "ai-ecoverse/typed-decision-fixtures@1",
               "source": {"repo": os.environ.get("RECIPE_SOURCE_REPO"), "revision": os.environ.get("RECIPE_SOURCE_REVISION")},
               "reference": f"PyTorch {torch.__version__} fp32 CPU, GemmaDecisionModel.forward / decide",
               "prompt": model.config.prompt, "temperatures": model.temperatures.tolist(), "fixtures": out},
              open(a.out, "w"))
    ok = [e for e in out if "error" not in e]
    print(f"{a.out}: {len(out)} requests ({len(out) - len(ok)} refused), {sum(len(e['texts']) for e in ok)} texts",
          flush=True)

    from runtime import Decider
    report = {}
    for v in a.variants + a.experiments:
        rt = Decider(a.export, v, a.onnx_dir if v in a.experiments else None)
        cos, dp, dl, changed, tokens = [], 0.0, 0.0, 0, True
        for e in ok:
            r = rt.decide(e["request"], e["workflow"])
            tokens &= r["ids"] == e["token_ids"] and r["texts"] == e["texts"]
            ref = np.asarray(e["embeddings"], np.float32)
            cos += list((r["embeddings"] * ref).sum(-1) / np.linalg.norm(r["embeddings"], axis=-1)
                        / np.linalg.norm(ref, axis=-1))
            for p, q, lg, lq in zip(e["probabilities"], r["probabilities"], e["logits"], r["logits"]):
                dp = max(dp, float(np.abs(np.asarray(p) - q[: len(p)]).max()))
                dl = max(dl, float(np.abs(np.asarray(lg) - lq[: len(lg)]).max()))
                changed += int(np.argmax(p) != np.argmax(q[: len(p)]))
        report[v] = {"texts_and_tokens_identical": bool(tokens), "min_cosine": float(min(cos)),
                     "mean_cosine": float(np.mean(cos)), "max_abs_dlogit": dl, "max_abs_dp": dp,
                     "answers_changed": changed, "questions": sum(len(e["questions"]) for e in ok)}
        print(v, report[v], flush=True)
    json.dump(report, open(a.results, "w"), indent=1)
    if any(not r["texts_and_tokens_identical"] for r in report.values()):
        raise SystemExit("the ONNX runtime's texts or token ids differ from the fixtures")


if __name__ == "__main__":
    sys.exit(main())
