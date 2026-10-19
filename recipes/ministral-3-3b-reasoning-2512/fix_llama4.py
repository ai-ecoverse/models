"""Make the decoders' position-dependent query scaling match transformers.

    python fix_llama4.py --out OUT --ckpt CKPT --results RESULTS

Ministral 3 multiplies its queries by 1 + beta * log(1 + floor(position / original_max_position_embeddings))
(transformers' get_llama_4_attn_scale; beta 0.1, original_max_position_embeddings 16384). Mistral's 3B ONNX
graphs compute it in /model/llama4_scaling with 262144 as the divisor, and in the fp16 graphs that constant is
stored as fp16, where it overflows to inf, so the scale is 1 at every position. fp16 also can't represent the
positions themselves above 2048, which would move the step by a few positions around each multiple of 16384.

This replaces the subgraph in every decoder_model_merged*.onnx with one computed in float32 from the checkpoint's
config, ending in a Cast to the graph's dtype under the same output name, so the 26 per-layer Mul nodes that use
it are untouched. Only the .onnx files change; the weights in .onnx_data don't. Each patched subgraph is then
evaluated at positions around the multiples of 16384 and compared with the transformers formula.
"""
import argparse, glob, json, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

P = "/model/llama4_scaling"
OLD = ["Cast_Pos", "Div", "Floor", "Add_Inner", "Log", "Mul_Beta", "Add_Final", "Unsqueeze_Broadcast"]
POSITIONS = np.array([[0, 1, 2047, 2049, 16383, 16384, 16385, 32767, 32768, 49151, 49152, 262143]], np.int64)


def subgraph(pos_in, out, dtype, beta, original_max):
    c = lambda name, value, dt=np.float32: helper.make_node(
        "Constant", [], [f"{P}/{name}"], value=numpy_helper.from_array(np.array(value, dt)), name=f"{P}/{name}")
    nodes = [
        c("max_position_fp32", original_max), c("one_fp32", 1.0), c("beta_fp32", beta), c("axes", [-1], np.int64),
        helper.make_node("Cast", [pos_in], [f"{P}/pos_fp32"], to=TensorProto.FLOAT, name=f"{P}/Cast_Pos"),
        helper.make_node("Div", [f"{P}/pos_fp32", f"{P}/max_position_fp32"], [f"{P}/div"], name=f"{P}/Div"),
        helper.make_node("Floor", [f"{P}/div"], [f"{P}/floor"], name=f"{P}/Floor"),
        helper.make_node("Add", [f"{P}/floor", f"{P}/one_fp32"], [f"{P}/inner"], name=f"{P}/Add_Inner"),
        helper.make_node("Log", [f"{P}/inner"], [f"{P}/log"], name=f"{P}/Log"),
        helper.make_node("Mul", [f"{P}/log", f"{P}/beta_fp32"], [f"{P}/mul"], name=f"{P}/Mul_Beta"),
        helper.make_node("Add", [f"{P}/mul", f"{P}/one_fp32"], [f"{P}/scale_fp32"], name=f"{P}/Add_Final"),
        helper.make_node("Unsqueeze", [f"{P}/scale_fp32", f"{P}/axes"], [f"{P}/scale3_fp32"], name=f"{P}/Unsqueeze_Broadcast"),
    ]
    if dtype == TensorProto.FLOAT:
        nodes.append(helper.make_node("Identity", [f"{P}/scale3_fp32"], [out], name=f"{P}/Output"))
    else:
        nodes.append(helper.make_node("Cast", [f"{P}/scale3_fp32"], [out], to=dtype, name=f"{P}/Output"))
    return nodes


def patch(path, beta, original_max):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    byname = {n.name: n for n in g.node}
    old = [byname.get(f"{P}/{k}") for k in OLD]
    if any(n is None for n in old):
        raise SystemExit(f"{path}: no {P} subgraph (already patched?)")
    pos_in = old[0].input[0]
    out = old[-1].output[0]
    dtype = old[0].attribute[0].i  # Cast_Pos's target type is the graph's compute dtype
    first = list(g.node).index(old[0])
    keep = [n for n in g.node if n not in old]
    new = subgraph(pos_in, out, dtype, beta, original_max)
    del g.node[:]
    g.node.extend(keep[:first] + new + keep[first:])
    # Constants the old subgraph used and nothing else uses any more.
    used = {i for n in g.node for i in n.input}
    nodes = [n for n in g.node if not (n.op_type == "Constant" and n.output[0] not in used and n.output[0].startswith("/model/constants/"))]
    del g.node[:]
    g.node.extend(nodes)
    onnx.save(m, path)
    return new, pos_in, out, dtype


def check(new, pos_in, out, dtype, beta, original_max):
    g = helper.make_graph(new, "llama4", [helper.make_tensor_value_info(pos_in, TensorProto.INT64, None)],
                          [helper.make_tensor_value_info(out, dtype, None)])
    got = ReferenceEvaluator(helper.make_model(g)).run(None, {pos_in: POSITIONS})[0].astype(np.float64)
    want = (1 + beta * np.log(1 + np.floor(POSITIONS / original_max)))[..., None]
    want = want.astype(np.float16 if dtype == TensorProto.FLOAT16 else np.float32).astype(np.float64)
    if not np.array_equal(got, want):
        raise SystemExit(f"scale mismatch: got {got.ravel().tolist()}, want {want.ravel().tolist()}")
    return dict(zip(POSITIONS.ravel().tolist(), got.ravel().tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    rope = json.load(open(os.path.join(a.ckpt, "config.json")))["text_config"]["rope_parameters"]
    beta, original_max = rope["llama_4_scaling_beta"], rope["original_max_position_embeddings"]
    report = {"beta": beta, "original_max_position_embeddings": original_max, "graphs": {}}
    for path in sorted(glob.glob(os.path.join(a.out, "onnx", "decoder_model_merged*.onnx"))):
        new, pos_in, out, dtype = patch(path, beta, original_max)
        scales = check(new, pos_in, out, dtype, beta, original_max)
        name = os.path.basename(path)
        report["graphs"][name] = {"dtype": TensorProto.DataType.Name(dtype), "scale": scales}
        print(f"{name}: scale at 16383 / 16384 / 32768 = {scales[16383]} / {scales[16384]:.6g} / {scales[32768]:.6g}", flush=True)
    json.dump(report, open(os.path.join(a.results, "llama4.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
