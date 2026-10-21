"""Lay out $OUT/onnx/ and build the variants from the fp32 export.

Encoder: fp32 (one .onnx_data sidecar), fp16 (Softmax/LayerNorm/reductions kept fp32), q8 (_quantized: dynamic
int8 on the constant-weight MatMuls), q4f16 (4-bit MatMulNBits, block 32, symmetric, then fp16). The decoder_joint
graph is small (~70 MB) and stays fp32 in every variant, written under each suffix so a variant is one suffix.

    python variants.py --raw raw --out $OUT/onnx --variants fp32 fp16 q8 q4f16
"""
import argparse, os, tempfile
import onnx

SUFFIX = {"fp32": "", "fp16": "_fp16", "q8": "_quantized", "q4": "_q4", "q4f16": "_q4f16"}
FP16_BLOCK = ["Softmax", "LayerNormalization", "ReduceMean", "Pow", "Sqrt"]

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variants", nargs="+", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def save(model, path):
    for f in (path, path + "_data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(path) + "_data", size_threshold=1024)


def load_shaped():
    m = onnx.load(shaped, load_external_data=False)
    onnx.load_external_data_for_model(m, a.out)
    return m


def to_fp16(m):
    from onnxruntime.transformers.float16 import convert_float_to_float16
    return convert_float_to_float16(m, keep_io_types=True, op_block_list=FP16_BLOCK, disable_shape_infer=True)


enc32 = f"{a.out}/encoder_model.onnx"
save(onnx.load(f"{a.raw}/encoder_model.onnx", load_external_data=True), enc32)
# value_info for the fp16 converter; the path API copes with >2 GB models (the in-memory one can't serialize them)
shaped = f"{a.raw}/encoder_shaped.onnx"
onnx.shape_inference.infer_shapes_path(enc32, shaped)

dj = onnx.load(f"{a.raw}/decoder_joint_model.onnx")
for v in a.variants:
    onnx.save_model(dj, f"{a.out}/decoder_joint_model{SUFFIX[v]}.onnx")

for v in a.variants:
    dst = f"{a.out}/encoder_model{SUFFIX[v]}.onnx"
    if v == "fp32":
        continue
    if v == "q8":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        with tempfile.TemporaryDirectory(dir=a.raw) as tmp:
            quantize_dynamic(enc32, f"{tmp}/q.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                             use_external_data_format=True, extra_options={"MatMulConstBOnly": True})
            save(onnx.load(f"{tmp}/q.onnx", load_external_data=True), dst)
    elif v == "fp16":
        save(to_fp16(load_shaped()), dst)
    elif v in ("q4", "q4f16"):
        from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
        q = MatMulNBitsQuantizer(load_shaped(), algo_config=DefaultWeightOnlyQuantConfig(
            block_size=32, is_symmetric=True, accuracy_level=4 if v == "q4" else None))
        q.process()
        save(to_fp16(q.model.model) if v == "q4f16" else q.model.model, dst)
    else:
        raise SystemExit(f"unknown variant {v}")
    print("built", v, flush=True)
