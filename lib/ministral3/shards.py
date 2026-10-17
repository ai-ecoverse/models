"""Save an ONNX model with its weights in external-data shards, transformers.js style.

    save_sharded(model, out_dir, "decoder_model_merged_q4f16.onnx")

writes out_dir/decoder_model_merged_q4f16.onnx plus decoder_model_merged_q4f16.onnx_data, _data_1, … Each shard
stays under max_bytes (default 2 GB: browsers can't read a larger file into one ArrayBuffer), and every tensor
starts at a 4 KiB-aligned offset. Returns the number of data files, the value transformers.js_config's
use_external_data_format expects.
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper

ALIGN = 4096


def _payload(t):
    if t.HasField("raw_data"):
        return t.raw_data
    return numpy_helper.to_array(t).tobytes()


def save_sharded(model, out_dir, name, max_bytes=2_000_000_000, min_external=1024):
    os.makedirs(out_dir, exist_ok=True)
    shards, files, offset, index = [], [], 0, 0

    def shard_name(i):
        return f"{name}_data" if i == 0 else f"{name}_data_{i}"

    f = open(os.path.join(out_dir, shard_name(0)), "wb")
    files.append(shard_name(0))
    for t in model.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            raise SystemExit(f"{t.name}: load the model with its external data before re-sharding")
        data = _payload(t)
        if len(data) < min_external:
            continue
        if len(data) > max_bytes:
            raise SystemExit(f"{t.name}: {len(data)} bytes is larger than one shard")
        start = -(-offset // ALIGN) * ALIGN
        if start + len(data) > max_bytes and offset > 0:
            f.close()
            index += 1
            f = open(os.path.join(out_dir, shard_name(index)), "wb")
            files.append(shard_name(index))
            start = 0
        f.seek(start)
        f.write(data)
        offset = start + len(data)
        for field in ("raw_data", "float_data", "int32_data", "int64_data", "double_data", "uint64_data", "string_data"):
            t.ClearField(field)
        del t.external_data[:]
        for k, v in (("location", shard_name(index)), ("offset", str(start)), ("length", str(len(data)))):
            e = t.external_data.add()
            e.key, e.value = k, v
        t.data_location = onnx.TensorProto.EXTERNAL
    f.close()
    onnx.save(model, os.path.join(out_dir, name))
    return len(files)


def external_files(out_dir, name):
    """Data files of a saved model, from its initializers' locations."""
    m = onnx.load(os.path.join(out_dir, name), load_external_data=False)
    return sorted({e.value for t in m.graph.initializer for e in t.external_data if e.key == "location"})


def dims(value):
    return [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]


def set_last_dim(value, n):
    value.type.tensor_type.shape.dim[-1].ClearField("dim_param")
    value.type.tensor_type.shape.dim[-1].dim_value = n


def to_numpy_dtype(data_type):
    return {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.UINT8: np.uint8, onnx.TensorProto.INT64: np.int64}[data_type]
