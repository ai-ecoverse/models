"""Export the decision head (with the per-type temperatures) as its own small ONNX graph, and decision_config.json.

    python head.py --src SRC --out OUT

onnx/decision_head.onnx takes the request's unique text embeddings and the index tensors of
`GemmaDecisionModel.forward` (render.build gives them), and returns the raw option logits and the calibrated
probabilities, exactly `GemmaDecisionModel.forward(...).logits / .probabilities`:

    text_embeds     float32 [N, 768]  L2-normalized embeddings (the encoder's sentence_embedding)
    state_index     int64   [Q]       row of each question's state
    question_index  int64   [Q]       row of each question's text
    option_index    int64   [Q, K]    rows of each question's options (padding: any valid row)
    option_mask     bool    [Q, K]    real options
    question_type   int64   [Q]       0 choice, 1 score, 2 noul
 -> logits          float32 [Q, K]    raw scores, padded options -1e4 (softmax of these is "T = 1")
    probabilities   float32 [Q, K]    softmax(logits / temperature[question_type]), padded options 0
"""
import argparse, json, os, sys
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

NAMES_IN = ["text_embeds", "state_index", "question_index", "option_index", "option_mask", "question_type"]


class Head(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.register_buffer("temperatures", model.temperatures.detach().clone().float())

    def forward(self, text_embeds, state_index, question_index, option_index, option_mask, question_type):
        # GemmaDecisionModel.forward after self.embed()
        logits = self.head(text_embeds[state_index], text_embeds[question_index], text_embeds[option_index],
                           question_type, option_mask).float()
        temps = self.temperatures[question_type].unsqueeze(-1)
        probabilities = (logits / temps).masked_fill(~option_mask, -1e4).softmax(-1)
        return logits, probabilities


def example(n_texts, q, k, seed):
    g = torch.Generator().manual_seed(seed)
    e = torch.nn.functional.normalize(torch.randn(n_texts, 768, generator=g), dim=-1)
    oi = torch.randint(0, n_texts, (q, k), generator=g)
    mask = torch.ones(q, k, dtype=torch.bool)
    mask[0, k - 1:] = False
    return (e, torch.randint(0, n_texts, (q,), generator=g), torch.randint(0, n_texts, (q,), generator=g), oi, mask,
            torch.randint(0, 3, (q,), generator=g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    model = AutoModel.from_pretrained(a.src, trust_remote_code=True, dtype=torch.float32).eval()
    head = Head(model).eval()
    os.makedirs(os.path.join(a.out, "onnx"), exist_ok=True)
    path = os.path.join(a.out, "onnx", "decision_head.onnx")
    with torch.no_grad():
        torch.onnx.export(head, example(9, 3, 4, 0), path, dynamo=False, opset_version=17,
                          input_names=NAMES_IN, output_names=["logits", "probabilities"],
                          dynamic_axes={"text_embeds": {0: "num_texts"}, "state_index": {0: "num_questions"},
                                        "question_index": {0: "num_questions"},
                                        "option_index": {0: "num_questions", 1: "max_options"},
                                        "option_mask": {0: "num_questions", 1: "max_options"},
                                        "question_type": {0: "num_questions"},
                                        "logits": {0: "num_questions", 1: "max_options"},
                                        "probabilities": {0: "num_questions", 1: "max_options"}})
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = {"logits": 0.0, "probabilities": 0.0}
    for seed, (n, q, k) in enumerate([(9, 3, 4), (40, 5, 9), (2, 1, 2), (130, 12, 20)]):
        x = example(n, q, k, seed + 1)
        with torch.no_grad():
            ref = [t.numpy() for t in head(*x)]
        got = sess.run(None, {name: t.numpy() for name, t in zip(NAMES_IN, x)})
        for name, r, g in zip(("logits", "probabilities"), ref, got):
            worst[name] = max(worst[name], float(np.abs(r - g).max()))
    print("decision_head.onnx vs PyTorch head (random inputs):", worst, flush=True)
    os.makedirs(a.results, exist_ok=True)
    json.dump(worst, open(os.path.join(a.results, "head.json"), "w"), indent=1)
    if worst["logits"] > 1e-4 or worst["probabilities"] > 1e-5:
        raise SystemExit("decision_head.onnx drifts from the PyTorch head")

    cfg = model.config
    json.dump({
        "format": "ai-ecoverse/typed-decision@1",
        "source": {"repo": os.environ.get("RECIPE_SOURCE_REPO"), "revision": os.environ.get("RECIPE_SOURCE_REVISION")},
        "question_types": list(cfg.question_types),
        "temperatures": [float(t) for t in model.temperatures.tolist()],
        "prompt": cfg.prompt,
        "pooling": {"mode": cfg.pooling_mode, "include_prompt": cfg.pooling_include_prompt, "normalize": True,
                    "embed_dim": cfg.embed_dim, "projections": cfg.projections},
        "encoder": {"graph": "onnx/model{suffix}.onnx", "output": "sentence_embedding",
                    "equivalent": "masked mean of last_hidden_state over attention_mask, then L2 normalization",
                    "media_inputs": ["image_features", "video_features", "audio_features"],
                    "media_input_shape": [0, 512]},
        "tokenizer": {"file": "tokenizer.json", "add_special_tokens": True, "padding_side": "right",
                      "pad_token_id": 0, "max_length": render.MAX_TOKENS, "truncation": "right"},
        "head": {"graph": "onnx/decision_head.onnx", "inputs": NAMES_IN, "outputs": ["logits", "probabilities"],
                 "option_pad_logit": render.OPTION_PAD},
        "noul_options": {"options": ["false", "true"], "descriptions": ["no", "yes"]},
    }, open(os.path.join(a.out, "decision_config.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
