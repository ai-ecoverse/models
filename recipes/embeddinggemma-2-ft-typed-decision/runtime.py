"""The ONNX runtime of the export, without PyTorch or transformers: what a JS runtime has to do, in Python.

render.build -> tokenizer.json (prompt + text, <bos> … <eos>, right padding) -> onnx/model{suffix}.onnx
(sentence_embedding) -> onnx/decision_head.onnx -> answers.

    rt = Decider(OUT, "q8")
    out = rt.decide({"state": ..., "questions": ...}, workflow="")
    out["answers"], out["probabilities"], out["logits"]
"""
import json, os, sys, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

SUFFIX = {"fp32": "", "q8": "_quantized", "fp16": "_fp16", "q4": "_q4", "q4f16": "_q4f16", "q8f16": "_q8f16"}


def softmax_masked(logits, mask):
    x = np.where(mask, logits, render.OPTION_PAD).astype(np.float64)
    x = np.exp(x - x.max(-1, keepdims=True))
    return (x / x.sum(-1, keepdims=True)).astype(np.float32)


class Decider:
    def __init__(self, out, variant="fp32", onnx_dir=None, threads=None):
        self.cfg = json.load(open(os.path.join(out, "decision_config.json")))
        self.tok = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        enc = os.path.join(onnx_dir or os.path.join(out, "onnx"), f"model{SUFFIX[variant]}.onnx")
        self.encoder = ort.InferenceSession(enc, so, providers=["CPUExecutionProvider"])
        self.head = ort.InferenceSession(os.path.join(out, "onnx", "decision_head.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.media = {i.name: np.zeros((0, 512), np.float32) for i in self.encoder.get_inputs()
                      if i.name not in ("input_ids", "attention_mask")}

    def tokenize(self, texts):
        ids = [e.ids for e in self.tok.encode_batch([self.cfg["prompt"] + t for t in texts], add_special_tokens=True)]
        cap = self.cfg["tokenizer"]["max_length"]
        return [i if len(i) <= cap else i[: cap - 1] + [i[-1]] for i in ids]   # keep <eos>, as transformers truncates

    def embed(self, ids, slack=1.25):
        """Embeddings of token id lists, in batches of similar length: padding to the request's longest text (the
        state, usually) would multiply the work, and a batch changes no result beyond float rounding (~2e-7)."""
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = np.empty((len(ids), self.cfg["pooling"]["embed_dim"]), np.float32)
        batch, real = [], 0
        for i in order + [None]:
            n = len(ids[i]) if i is not None else 0
            if batch and (i is None or (len(batch) + 1) * n > slack * (real + n)):
                out[batch] = self.embed_batch([ids[j] for j in batch])
                batch, real = [], 0
            if i is not None:
                batch.append(i)
                real += n
        return out

    def embed_batch(self, ids):
        t = max(len(i) for i in ids)
        input_ids = np.zeros((len(ids), t), np.int64)
        mask = np.zeros((len(ids), t), np.int64)
        for r, i in enumerate(ids):
            input_ids[r, : len(i)] = i
            mask[r, : len(i)] = 1
        return self.encoder.run(["sentence_embedding"], {"input_ids": input_ids, "attention_mask": mask,
                                                         **self.media})[0]

    def decide(self, request, workflow=""):
        t0 = time.perf_counter()
        texts, lay = render.build(request, workflow)
        ids = self.tokenize(texts)
        t1 = time.perf_counter()
        emb = self.embed(ids)
        t2 = time.perf_counter()
        feeds = {"text_embeds": emb.astype(np.float32),
                 "state_index": np.array(lay["state_index"], np.int64),
                 "question_index": np.array(lay["question_index"], np.int64),
                 "option_index": np.array(lay["option_index"], np.int64),
                 "option_mask": np.array(lay["option_mask"], bool),
                 "question_type": np.array(lay["question_type"], np.int64)}
        logits, probs = self.head.run(["logits", "probabilities"], feeds)
        t3 = time.perf_counter()
        return {"texts": texts, "layout": lay, "ids": ids, "embeddings": emb, "logits": logits,
                "probabilities": probs, "raw_probabilities": softmax_masked(logits, feeds["option_mask"]),
                "answers": render.answers(lay["specs"], probs),
                "ms": {"render_tokenize": (t1 - t0) * 1e3, "encoder": (t2 - t1) * 1e3, "head": (t3 - t2) * 1e3,
                       "total": (t3 - t0) * 1e3}}
