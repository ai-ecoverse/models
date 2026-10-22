import argparse
import json
import logging
import math
import os
import re
import shutil
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

D_MODEL = 512
HEADS = 8
HEAD_DIM = 64
LDIM = 32
MIMI_CONTEXT = 250
FRAME = 1920
OPSET = 17
GRAPHS = ("text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder")
FLOW_LM = ("flow_lm_main", "flow_lm_flow")
VARIANTS = {"quantized": 8, "q4": 4, "fp16": None}
DEFAULT_VOICE = "easton"


def load_weights(path):
    out = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).to(torch.float32)
    return out


def frozen(t):
    return torch.nn.Parameter(t, requires_grad=False)


def rope_tables(positions, max_period=10000.0):
    half = HEAD_DIM // 2
    inv = 1.0 / (max_period ** (torch.arange(half, dtype=torch.float32) / half))
    ang = positions.to(torch.float32)[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)


def rope(x, cos, sin):
    b, h, t, d = x.shape
    x = x.reshape(b, h, t, d // 2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).reshape(b, h, t, d)


def additive_mask(ok):
    return torch.where(
        ok,
        torch.zeros((), dtype=torch.float32),
        torch.full((), float("-inf"), dtype=torch.float32),
    )


class Layer(torch.nn.Module):
    def __init__(self, w, prefix, scale):
        super().__init__()
        g = lambda n: frozen(w[f"{prefix}.{n}"])
        self.n1w, self.n1b = g("norm1.weight"), g("norm1.bias")
        self.n2w, self.n2b = g("norm2.weight"), g("norm2.bias")
        self.in_proj = g("self_attn.in_proj.weight")
        self.out_proj = g("self_attn.out_proj.weight")
        self.l1 = g("linear1.weight")
        self.l2 = g("linear2.weight")
        self.scale = scale
        if scale:
            self.s1 = g("layer_scale_1.scale")
            self.s2 = g("layer_scale_2.scale")

    def qkv(self, x, cos, sin):
        b, t, _ = x.shape
        h = F.layer_norm(x, (D_MODEL,), self.n1w, self.n1b, 1e-5)
        p = (h @ self.in_proj.t()).reshape(b, t, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        return rope(p[0], cos, sin), rope(p[1], cos, sin), p[2]

    def finish(self, x, q, k, v, mask):
        b, _, t, _ = q.shape
        a = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(HEAD_DIM)) + mask
        a = torch.softmax(a, dim=-1) @ v
        a = a.transpose(1, 2).reshape(b, t, D_MODEL) @ self.out_proj.t()
        if self.scale:
            a = a * self.s1
        x = x + a
        h = F.layer_norm(x, (D_MODEL,), self.n2w, self.n2b, 1e-5)
        f = F.gelu(h @ self.l1.t()) @ self.l2.t()
        if self.scale:
            f = f * self.s2
        return x + f


class FlowLMMain(torch.nn.Module):
    def __init__(self, w, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Layer(w, f"flow_lm.transformer.layers.{i}", False) for i in range(layers)]
        )
        self.onw = frozen(w["flow_lm.out_norm.weight"])
        self.onb = frozen(w["flow_lm.out_norm.bias"])
        self.eosw = frozen(w["flow_lm.out_eos.weight"])
        self.eosb = frozen(w["flow_lm.out_eos.bias"])

    def forward(self, inputs_embeds, *past):
        t = inputs_embeds.shape[1]
        p = past[0].shape[2]
        pos = torch.arange(t, dtype=torch.int64) + p
        cos, sin = rope_tables(pos)
        keys = torch.arange(p + t, dtype=torch.int64)
        mask = additive_mask(keys[None, :] <= pos[:, None])
        x = inputs_embeds
        present = []
        for i, layer in enumerate(self.layers):
            q, k, v = layer.qkv(x, cos, sin)
            k = torch.cat([past[2 * i], k], dim=2)
            v = torch.cat([past[2 * i + 1], v], dim=2)
            present += [k, v]
            x = layer.finish(x, q, k, v, mask)
        last = F.layer_norm(x[:, -1, :], (D_MODEL,), self.onw, self.onb, 1e-5)
        eos = last @ self.eosw.t() + self.eosb
        return (last, eos, *present)


def rms(x, alpha, eps=1e-5):
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * alpha


class FlowHead(torch.nn.Module):
    def __init__(self, w, depth, cond):
        super().__init__()
        p = "flow_lm.flow_net"
        g = lambda n: frozen(w[f"{p}.{n}"])
        self.te = []
        for i in range(2):
            te = [
                g(f"time_embed.{i}.freqs"),
                g(f"time_embed.{i}.mlp.0.weight"),
                g(f"time_embed.{i}.mlp.0.bias"),
                g(f"time_embed.{i}.mlp.2.weight"),
                g(f"time_embed.{i}.mlp.2.bias"),
                g(f"time_embed.{i}.mlp.3.alpha"),
            ]
            for j, t in enumerate(te):
                self.register_parameter(f"te{i}_{j}", t)
            self.te.append(te)
        self.cw, self.cb = g("cond_embed.weight"), g("cond_embed.bias")
        self.iw, self.ib = g("input_proj.weight"), g("input_proj.bias")
        self.blocks = []
        for i in range(depth):
            blk = [
                g(f"res_blocks.{i}.in_ln.weight"),
                g(f"res_blocks.{i}.in_ln.bias"),
                g(f"res_blocks.{i}.mlp.0.weight"),
                g(f"res_blocks.{i}.mlp.0.bias"),
                g(f"res_blocks.{i}.mlp.2.weight"),
                g(f"res_blocks.{i}.mlp.2.bias"),
                g(f"res_blocks.{i}.adaLN_modulation.1.weight"),
                g(f"res_blocks.{i}.adaLN_modulation.1.bias"),
            ]
            for j, t in enumerate(blk):
                self.register_parameter(f"b{i}_{j}", t)
            self.blocks.append(blk)
        self.fw, self.fb = g("final_layer.linear.weight"), g("final_layer.linear.bias")
        self.faw = g("final_layer.adaLN_modulation.1.weight")
        self.fab = g("final_layer.adaLN_modulation.1.bias")
        self.liw = frozen(w["flow_lm.input_linear.weight"])
        self.cond = cond

    def time(self, te, t):
        freqs, w1, b1, w2, b2, alpha = te
        a = t * freqs
        e = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        return rms(F.silu(e @ w1.t() + b1) @ w2.t() + b2, alpha)

    def forward(self, hidden, noise, duration_delta):
        s = torch.zeros((hidden.shape[0], 1), dtype=torch.float32)
        t = torch.ones((hidden.shape[0], 1), dtype=torch.float32)
        tc = (self.time(self.te[0], s) + self.time(self.te[1], t)) * 0.5
        sy = F.silu(tc + hidden @ self.cw.t() + self.cb)
        x = noise @ self.iw.t() + self.ib
        for lw, lb, w1, b1, w2, b2, aw, ab in self.blocks:
            ada = sy @ aw.t() + ab
            shift, scale, gate = ada[:, :D_MODEL], ada[:, D_MODEL : 2 * D_MODEL], ada[:, 2 * D_MODEL :]
            h = F.layer_norm(x, (D_MODEL,), lw, lb, 1e-6) * (1 + scale) + shift
            h = F.silu(h @ w1.t() + b1) @ w2.t() + b2
            x = x + gate * h
        ada = sy @ self.faw.t() + self.fab
        shift, scale = ada[:, :D_MODEL], ada[:, D_MODEL:]
        h = F.layer_norm(x, (D_MODEL,), None, None, 1e-6) * (1 + scale) + shift
        latent = noise + (h @ self.fw.t() + self.fb)
        nxt = latent @ self.liw.t() + self.cond(duration_delta)
        return latent[:, None, :], nxt[:, None, :]


class TextConditioner(torch.nn.Module):
    def __init__(self, w, bos_embed, cond):
        super().__init__()
        self.table = frozen(w["flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight"])
        self.bos = frozen(bos_embed)
        self.cond = cond

    def forward(self, input_ids, duration_delta):
        bos = self.bos + self.cond(duration_delta)
        return self.table[input_ids], bos + torch.zeros_like(self.table[input_ids[:, :1]])


class Conditions(torch.nn.Module):
    def __init__(self, w, cfg):
        super().__init__()
        p = "flow_lm.condition_provider.conditioners"
        static = torch.zeros(D_MODEL)
        self.period = None
        for c in cfg["conditioners"]:
            name = c["name"]
            proj = w.get(f"{p}.{name}.output_proj.weight")
            if c["type"] == "lut":
                idx = c["lut"]["possible_values"].index({"num_speakers": "1"}.get(name, "0"))
                e = w[f"{p}.{name}.embed.weight"][idx]
                static = static + (e if proj is None else proj @ e)
            elif name == "duration_delta":
                cc = c["continuous"]
                half = cc["dim"] // 2
                i = np.arange(half, dtype=np.float32) / np.float32(half - 1)
                self.scale = float(cc["scale_factor"])
                self.period = frozen(torch.from_numpy(np.float32(cc["max_period"]) ** i))
                self.proj = None if proj is None else frozen(proj)
            else:
                raise SystemExit(f"continuous conditioner {name} is not supported")
        if self.period is None:
            raise SystemExit("no duration_delta conditioner")
        self.static = frozen(static)

    def forward(self, duration_delta):
        phases = (duration_delta * self.scale) / self.period
        e = torch.cat([torch.cos(phases), torch.sin(phases)])
        if self.proj is not None:
            e = e @ self.proj.t()
        return self.static + e


class MimiDecoder(torch.nn.Module):
    def __init__(self, w, cfg):
        super().__init__()
        m = cfg["mimi"]
        self.ratios = m["ratios"]
        self.std = frozen(w["flow_lm.emb_std"])
        self.mean = frozen(w["flow_lm.emb_mean"])
        self.qw = frozen(w["mimi.quantizer.output_proj.weight"])
        self.uw = frozen(w["mimi.upsample.convtr.convtr.weight"])
        self.layers = torch.nn.ModuleList(
            [
                Layer(w, f"mimi.decoder_transformer.transformer.layers.{i}", True)
                for i in range(m["transformer_num_layers"])
            ]
        )
        self.params = {}
        for k, v in w.items():
            if k.startswith("mimi.decoder.model."):
                name = k.replace(".", "_")
                self.register_parameter(name, frozen(v))
                self.params[k] = name

    def p(self, k):
        return getattr(self, self.params[k])

    def conv(self, x, prev, idx):
        w = self.p(f"mimi.decoder.model.{idx}.conv.weight")
        b = self.p(f"mimi.decoder.model.{idx}.conv.bias")
        if prev is None:
            return F.conv1d(x, w, b), None
        x = torch.cat([prev, x], dim=2)
        return F.conv1d(x, w, b), x[:, :, x.shape[2] - prev.shape[2] :]

    def convtr(self, x, partial, w, b, stride, groups=1):
        y = F.conv_transpose1d(x, w, None, stride=stride, groups=groups)
        if b is not None:
            y = y + b[None, :, None]
        pt = partial.shape[2]
        head = y[:, :, :pt] + partial
        tail = y[:, :, y.shape[2] - pt :]
        if b is not None:
            tail = tail - b[None, :, None]
        return torch.cat([head, y[:, :, pt : y.shape[2] - pt]], dim=2), tail

    def forward(self, latent, offset, *state):
        state = list(state)
        out = []
        x = (latent * self.std + self.mean).transpose(1, 2)
        x = F.conv1d(x, self.qw)
        x, up = self.convtr(x, state.pop(0), self.uw, None, 16, groups=D_MODEL)
        out.append(up)
        x = x.transpose(1, 2)
        t = x.shape[1]
        pos = torch.arange(t, dtype=torch.int64) + offset
        cos, sin = rope_tables(pos)
        kpos = torch.arange(MIMI_CONTEXT + t, dtype=torch.int64) + offset - MIMI_CONTEXT
        mask = additive_mask(
            (kpos[None, :] >= 0)
            & (kpos[None, :] >= pos[:, None] - MIMI_CONTEXT)
            & (kpos[None, :] <= pos[:, None])
        )
        kvs = []
        for layer in self.layers:
            pk, pv = state.pop(0), state.pop(0)
            q, k, v = layer.qkv(x, cos, sin)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            kvs += [k[:, :, k.shape[2] - MIMI_CONTEXT :], v[:, :, v.shape[2] - MIMI_CONTEXT :]]
            x = layer.finish(x, q, k, v, mask)
        out += kvs
        z, s = self.conv(x.transpose(1, 2), state.pop(0), 0)
        out.append(s)
        idx = 1
        for r in self.ratios:
            z = F.elu(z)
            w = self.p(f"mimi.decoder.model.{idx + 1}.convtr.weight")
            b = self.p(f"mimi.decoder.model.{idx + 1}.convtr.bias")
            z, s = self.convtr(z, state.pop(0), w, b, r)
            out.append(s)
            idx += 2
            v, s = self.conv(F.elu(z), state.pop(0), f"{idx}.block.1")
            out.append(s)
            v, _ = self.conv(F.elu(v), None, f"{idx}.block.3")
            z = z + v
            idx += 1
        z, s = self.conv(F.elu(z), state.pop(0), idx + 1)
        out.append(s)
        return (z, offset + t, *out)

    def state_shapes(self):
        shapes = [("upsample", (1, D_MODEL, 16))]
        for i in range(len(self.layers)):
            shapes.append((f"key_{i}", (1, HEADS, MIMI_CONTEXT, HEAD_DIM)))
            shapes.append((f"value_{i}", (1, HEADS, MIMI_CONTEXT, HEAD_DIM)))
        shapes.append(("conv_in", (1, D_MODEL, 6)))
        ch = D_MODEL
        for n, r in enumerate(self.ratios):
            ch //= 2
            shapes.append((f"convtr_{n}", (1, ch, r)))
            shapes.append((f"res_{n}", (1, ch, 2)))
        shapes.append(("conv_out", (1, 64, 2)))
        return shapes

    def init_state(self):
        return [torch.zeros(s) for _, s in self.state_shapes()]


class Phonon:
    def __init__(self, model_dir):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        w = load_weights(os.path.join(model_dir, "model.safetensors"))
        fl = self.cfg["flow_lm"]
        self.layers = fl["num_layers"]
        self.cond = Conditions(w, self.cfg).eval()
        self.main = FlowLMMain(w, self.layers).eval()
        self.flow = FlowHead(w, fl["flow_depth"], self.cond).eval()
        bos = w["flow_lm.bos_emb"] @ w["flow_lm.input_linear.weight"].t()
        self.text = TextConditioner(w, bos[None, None, :], self.cond).eval()
        self.mimi = MimiDecoder(w, self.cfg).eval()
        self.speaker = w["flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight"]

    def voice(self, path):
        with safe_open(path, "pt") as f:
            lat = f.get_tensor("speaker_wavs").to(torch.float32)
        return lat.transpose(1, 2) @ self.speaker.t()


def export(ph, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    past = [torch.zeros((1, HEADS, 3, HEAD_DIM)) for _ in range(2 * ph.layers)]
    past_names, present_names = [], []
    axes = {"inputs_embeds": {1: "sequence"}}
    for i in range(ph.layers):
        for kv in ("key", "value"):
            past_names.append(f"past_key_values.{i}.{kv}")
            present_names.append(f"present.{i}.{kv}")
            axes[f"past_key_values.{i}.{kv}"] = {2: "past_sequence"}
            axes[f"present.{i}.{kv}"] = {2: "total_sequence"}
    common = dict(opset_version=OPSET, dynamo=False)
    warnings.filterwarnings("ignore", message="You are using the legacy TorchScript-based ONNX export")
    with torch.no_grad():
        torch.onnx.export(
            ph.main,
            (torch.zeros((1, 2, D_MODEL)), *past),
            os.path.join(out_dir, "flow_lm_main.onnx"),
            input_names=["inputs_embeds", *past_names],
            output_names=["hidden", "eos_logit", *present_names],
            dynamic_axes=axes,
            **common,
        )
        torch.onnx.export(
            ph.flow,
            (torch.zeros((1, D_MODEL)), torch.zeros((1, LDIM)), torch.zeros((1,))),
            os.path.join(out_dir, "flow_lm_flow.onnx"),
            input_names=["hidden", "noise", "duration_delta"],
            output_names=["latent", "next_embeds"],
            **common,
        )
        torch.onnx.export(
            ph.text,
            (torch.zeros((1, 4), dtype=torch.int64), torch.zeros((1,))),
            os.path.join(out_dir, "text_conditioner.onnx"),
            input_names=["input_ids", "duration_delta"],
            output_names=["inputs_embeds", "bos_embeds"],
            dynamic_axes={"input_ids": {1: "sequence"}, "inputs_embeds": {1: "sequence"}},
            **common,
        )
        names = [n for n, _ in ph.mimi.state_shapes()]
        torch.onnx.export(
            ph.mimi,
            (torch.zeros((1, 1, LDIM)), torch.tensor(0, dtype=torch.int64), *ph.mimi.init_state()),
            os.path.join(out_dir, "mimi_decoder.onnx"),
            input_names=["latent", "offset", *[f"state.{n}" for n in names]],
            output_names=["audio", "next_offset", *[f"next_state.{n}" for n in names]],
            dynamic_axes={"latent": {1: "frames"}, "audio": {2: "samples"}},
            **common,
        )


def dedupe(model):
    outs, names, keep = set(), {}, []
    for node in model.graph.node:
        if any(o in outs for o in node.output):
            continue
        outs.update(node.output)
        if node.name in names:
            names[node.name] += 1
            node.name = f"{node.name}_{names[node.name]}"
        else:
            names[node.name] = 0
        keep.append(node)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return model


def quantize(onnx_dir, kind):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    bits = VARIANTS[kind]
    for name in GRAPHS:
        src = os.path.join(onnx_dir, f"{name}.onnx")
        dst = os.path.join(onnx_dir, f"{name}_{kind}.onnx")
        cond = [n.name for n in onnx.load(src).graph.node if n.name.startswith("/cond/")]
        if bits and name in FLOW_LM:
            q = MatMulNBitsQuantizer(
                onnx.load(src), block_size=32, is_symmetric=True, bits=bits, accuracy_level=4, nodes_to_exclude=cond
            )
            q.process()
            q.model.save_model_to_file(dst, False)
        elif bits is None:
            blocked = list(DEFAULT_OP_BLOCK_LIST) + ["Sin", "Cos", "Range"]
            m = convert_float_to_float16(
                onnx.load(src), keep_io_types=True, op_block_list=blocked, node_block_list=cond
            )
            onnx.save(dedupe(m), dst)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def english_voices(voices_dir, language="en"):
    out = []
    for line in open(os.path.join(voices_dir, "voices.jsonl")):
        if line.strip():
            e = json.loads(line)
            if e["language"] == language:
                out.append(e)
    return out


def write_voices(ph, voices_dir, out_dir):
    os.makedirs(os.path.join(out_dir, "voices"), exist_ok=True)
    entries = []
    for e in english_voices(voices_dir):
        emb = ph.voice(os.path.join(voices_dir, e["filename"]))[0].numpy().astype(np.float32)
        name = slug(e["name"])
        if any(x["name"] == name for x in entries):
            raise SystemExit(f"two English voices are both called {name}")
        emb.tofile(os.path.join(out_dir, "voices", f"{name}.bin"))
        entries.append(
            {
                "name": name,
                "display_name": e["name"],
                "voice_id": e["voice_id"],
                "locale": e.get("locale"),
                "gender": e.get("gender"),
                "frames": int(emb.shape[0]),
            }
        )
    if not any(x["name"] == DEFAULT_VOICE for x in entries):
        raise SystemExit(f"no {DEFAULT_VOICE} voice among the English voices")
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", required=True)
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()
    if a.list_voices:
        for e in english_voices(a.voices):
            print(slug(e["name"]), e["filename"])
        return
    if not a.model or not a.out:
        ap.error("--model and --out are required to export")
    ph = Phonon(a.model)
    onnx_dir = os.path.join(a.out, "onnx")
    export(ph, onnx_dir)
    for kind in VARIANTS:
        quantize(onnx_dir, kind)
    entries = write_voices(ph, a.voices, a.out)
    cfg = dict(ph.cfg)
    cfg["model_type"] = "phonon"
    cfg["onnx"] = {
        "frame_samples": FRAME,
        "mimi_context": MIMI_CONTEXT,
        "num_layers": ph.layers,
        "num_heads": HEADS,
        "head_dim": HEAD_DIM,
        "latent_dim": LDIM,
        "mimi_state": [[n, list(s)] for n, s in ph.mimi.state_shapes()],
        "voices": entries,
        "default_voice": DEFAULT_VOICE,
        "inputs": {"duration_delta": {"default": 0}},
    }
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    shutil.copyfile(os.path.join(a.model, "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))


if __name__ == "__main__":
    main()
