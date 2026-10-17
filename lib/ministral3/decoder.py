"""Export a Ministral 3 decoder with onnxruntime-genai's model builder, in Mistral's transformers.js layout.

    python decoder.py --ckpt CKPT --variant q4f16 --out OUT --work WORK [--layers N]

The builder (Mistral3TextModel, exclude_embeds) writes the text decoder with inputs_embeds, attention_mask,
position_ids and past_key_values.N.{key,value} in, logits and present.N.{key,value} out, the same graph shape as
Mistral's own Ministral-3-3B-Instruct-2512-ONNX decoders (GroupQueryAttention, MatMulNBits, block 32, accuracy
level 4). Two things are added afterwards:

- Ministral 3's position-dependent query scaling (transformers' get_llama_4_attn_scale): queries are multiplied by
  1 + beta * log(1 + floor(position / original_max_position_embeddings)). The builder doesn't emit it. A scalar
  per position commutes with the rotary rotation, so it is applied to the query part of the packed QKV projection
  before attention. Positions come from attention_mask (its running sum minus one, last sequence_length entries),
  as GroupQueryAttention derives its rotary positions, so the graph keeps the builder's inputs (no position_ids).
  Mistral's 3B graphs divide by 262144 instead of 16384 (and in fp16 that constant is inf), so their scaling never
  applies; this one follows transformers.
- q4f16: inputs_embeds and logits stay float32, as in Mistral's q4f16 decoder, with fp16 inside and in the KV cache.

The result is re-saved as onnx/decoder_model_merged_<variant>.onnx with 2 GB data shards.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shards import save_sharded  # noqa: E402

SUFFIX = {"q4f16": "_q4f16", "q4": "_q4", "fp16": "_fp16", "fp32": ""}
BUILD = {  # precision, execution provider
    "q4f16": ("int4", "webgpu"),
    "q4": ("int4", "cpu"),
    "fp16": ("fp16", "webgpu"),
    "fp32": ("fp32", "cpu"),
}


def build(ckpt, variant, work, layers=None):
    precision, ep = BUILD[variant]
    out = os.path.join(work, f"genai-{variant}")
    if os.path.exists(os.path.join(out, "model.onnx")) and os.path.exists(os.path.join(out, "genai_config.json")):
        print(f"reusing {out}", flush=True)
        return out
    extra = ["exclude_embeds=true", "block_size=32", "accuracy_level=4"]
    if layers:
        extra.append(f"num_hidden_layers={layers}")
    cmd = [sys.executable, "-m", "onnxruntime_genai.models.builder", "-i", ckpt, "-o", out, "-p", precision, "-e", ep,
           "-c", os.path.join(work, "genai-cache"), "--extra_options", *extra]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def text_config(ckpt):
    c = json.load(open(os.path.join(ckpt, "config.json")))
    return c.get("text_config", c)


def add_query_scaling(model, beta, original_max, head_dim):
    """Multiply every GroupQueryAttention's query by the llama-4 position scale."""
    g = model.graph
    if not beta:
        return 0
    p = "/model/llama4_scaling"
    c = lambda name, value, dt=np.float32: numpy_helper.from_array(np.array(value, dt), f"{p}/{name}")
    g.initializer.extend([c("max", original_max), c("one", 1.0), c("beta", beta), c("axes", [-1], np.int64),
                          c("one_i", 1, np.int64), c("axis1", [1], np.int64), c("end", [np.iinfo(np.int64).max], np.int64)])
    if "position_ids" in {i.name for i in g.input}:
        nodes, positions = [], "position_ids"
    else:
        positions = f"{p}/positions"
        nodes = [
            helper.make_node("CumSum", ["attention_mask", f"{p}/one_i"], [f"{p}/cumsum"], name=f"{p}/CumSum"),
            helper.make_node("Sub", [f"{p}/cumsum", f"{p}/one_i"], [f"{p}/all_positions"], name=f"{p}/Sub"),
            helper.make_node("Shape", ["inputs_embeds"], [f"{p}/seq_len"], start=1, end=2, name=f"{p}/SeqLen"),
            helper.make_node("Neg", [f"{p}/seq_len"], [f"{p}/start"], name=f"{p}/Neg"),
            helper.make_node("Slice", [f"{p}/all_positions", f"{p}/start", f"{p}/end", f"{p}/axis1"], [positions], name=f"{p}/Slice"),
        ]
    nodes += [
        helper.make_node("Cast", [positions], [f"{p}/pos"], to=TensorProto.FLOAT, name=f"{p}/Cast"),
        helper.make_node("Div", [f"{p}/pos", f"{p}/max"], [f"{p}/div"], name=f"{p}/Div"),
        helper.make_node("Floor", [f"{p}/div"], [f"{p}/floor"], name=f"{p}/Floor"),
        helper.make_node("Add", [f"{p}/floor", f"{p}/one"], [f"{p}/inner"], name=f"{p}/Add_Inner"),
        helper.make_node("Log", [f"{p}/inner"], [f"{p}/log"], name=f"{p}/Log"),
        helper.make_node("Mul", [f"{p}/log", f"{p}/beta"], [f"{p}/mul"], name=f"{p}/Mul_Beta"),
        helper.make_node("Add", [f"{p}/mul", f"{p}/one"], [f"{p}/scale"], name=f"{p}/Add_Final"),
        helper.make_node("Unsqueeze", [f"{p}/scale", f"{p}/axes"], [f"{p}/scale3"], name=f"{p}/Unsqueeze"),
    ]
    io16 = any(i.type.tensor_type.elem_type == TensorProto.FLOAT16 for i in g.input if i.name.startswith("past_key_values"))
    if io16:
        nodes.append(helper.make_node("Cast", [f"{p}/scale3"], [f"{p}/scale3_16"], to=TensorProto.FLOAT16, name=f"{p}/Cast16"))
    scale = f"{p}/scale3_16" if io16 else f"{p}/scale3"
    out, count = [], 0
    for n in g.node:
        if n.op_type == "GroupQueryAttention":
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            q = n.input[0]
            packed = len(n.input) < 3 or not n.input[1]
            factor = scale
            if packed:
                # One QKV tensor: scale only its first num_heads * head_size channels.
                heads, kv = attrs["num_heads"], attrs["kv_num_heads"]
                mask = np.concatenate([np.ones(heads * head_dim), np.zeros(2 * kv * head_dim)]).astype(np.float16 if io16 else np.float32)
                mname = f"{n.name}/llama4_mask"
                g.initializer.append(numpy_helper.from_array(mask, mname))
                one = f"{p}/one16" if io16 else f"{p}/one"
                if io16 and not any(t.name == one for t in g.initializer):
                    g.initializer.append(numpy_helper.from_array(np.array(1.0, np.float16), one))
                out += [
                    helper.make_node("Sub", [scale, one], [f"{n.name}/llama4_sub"], name=f"{n.name}/llama4_sub"),
                    helper.make_node("Mul", [f"{n.name}/llama4_sub", mname], [f"{n.name}/llama4_masked"], name=f"{n.name}/llama4_masked"),
                    helper.make_node("Add", [f"{n.name}/llama4_masked", one], [f"{n.name}/llama4_factor"], name=f"{n.name}/llama4_factor"),
                ]
                factor = f"{n.name}/llama4_factor"
            scaled = f"{n.name}/llama4_scaling/output_0"
            out.append(helper.make_node("Mul", [q, factor], [scaled], name=f"{n.name}/llama4_scaling"))
            n.input[0] = scaled
            count += 1
        out.append(n)
    del g.node[:]
    g.node.extend(nodes + out)
    return count


def float_io(model):
    """Make inputs_embeds and logits float32 when the builder made them fp16."""
    g = model.graph
    changed = []
    for i in g.input:
        if i.name == "inputs_embeds" and i.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inner = "inputs_embeds_fp16"
            for n in g.node:
                for k, x in enumerate(n.input):
                    if x == "inputs_embeds":
                        n.input[k] = inner
            g.node.insert(0, helper.make_node("Cast", ["inputs_embeds"], [inner], to=TensorProto.FLOAT16, name="/model/inputs_embeds/Cast"))
            i.type.tensor_type.elem_type = TensorProto.FLOAT
            changed.append("inputs_embeds")
    for o in g.output:
        if o.name == "logits" and o.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inner = "logits_fp16"
            for n in g.node:
                for k, x in enumerate(n.output):
                    if x == "logits":
                        n.output[k] = inner
            g.node.append(helper.make_node("Cast", [inner], ["logits"], to=TensorProto.FLOAT, name="/model/logits/Cast"))
            o.type.tensor_type.elem_type = TensorProto.FLOAT
            changed.append("logits")
    return changed


def check_io(model, layers):
    ins = {i.name for i in model.graph.input}
    outs = {o.name for o in model.graph.output}
    want_in = {"inputs_embeds", "attention_mask"} | {f"past_key_values.{i}.{k}" for i in range(layers) for k in ("key", "value")}
    ins.discard("position_ids")
    want_out = {"logits"} | {f"present.{i}.{k}" for i in range(layers) for k in ("key", "value")}
    if ins != want_in or outs != want_out:
        raise SystemExit(f"unexpected decoder I/O: extra in {sorted(ins - want_in)[:4]}, missing in {sorted(want_in - ins)[:4]}, "
                         f"extra out {sorted(outs - want_out)[:4]}, missing out {sorted(want_out - outs)[:4]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=list(BUILD))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--layers", type=int, help="export only the first N layers (for testing)")
    a = ap.parse_args()
    tc = text_config(a.ckpt)
    head_dim = tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
    built = build(a.ckpt, a.variant, a.work, a.layers)
    model = onnx.load(os.path.join(built, "model.onnx"), load_external_data=True)
    rope = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    n = add_query_scaling(model, rope.get("llama_4_scaling_beta"), rope.get("original_max_position_embeddings"), head_dim)
    changed = float_io(model)
    check_io(model, a.layers or tc["num_hidden_layers"])
    name = f"decoder_model_merged{SUFFIX[a.variant]}.onnx"
    files = save_sharded(model, os.path.join(a.out, "onnx"), name)
    print(json.dumps({"file": name, "data_files": files, "llama4_scaled_attention": n, "float_io": changed}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
