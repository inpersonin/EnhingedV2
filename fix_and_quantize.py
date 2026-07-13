"""
fix_and_quantize.py

Converts model.onnx (fp16 weights) → fp32 → int8 quantized.

The fp16 source model breaks ONNX dynamic quantization because
the DequantizeLinear operator expects fp32 scales. This script
first casts all fp16 initializer tensors to fp32, then quantizes
only the MatMul and Gemm ops to int8.

Result: model_quant.onnx that loads cleanly in ONNX Runtime.
"""

import onnx
import numpy as np
from onnx import numpy_helper, TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC  = "model.onnx"
MID  = "model_fp32.onnx"
DST  = "model_quant.onnx"

# ── Step 1: load and cast fp16 → fp32 ──────────────────────────────────────
print(f"Loading {SRC} ...")
model = onnx.load(SRC)

converted = 0
for init in model.graph.initializer:
    if init.data_type == TensorProto.FLOAT16:
        arr = numpy_helper.to_array(init).astype(np.float32)
        new_init = numpy_helper.from_array(arr, name=init.name)
        init.CopyFrom(new_init)
        converted += 1

# Update graph input/output types as well
for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
    if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT

print(f"Converted {converted} fp16 tensors to fp32.")
onnx.checker.check_model(model)
onnx.save(model, MID)
print(f"Saved fp32 model to {MID}  ({round(onnx.external_data_helper.ExternalDataInfo(model.graph.node[0]) if False else __import__('os').path.getsize(MID)/1024/1024, 1)} MB)")

# ── Step 2: dynamic quantization (MatMul + Gemm only → int8) ───────────────
print(f"\nQuantizing {MID} → {DST} ...")
quantize_dynamic(
    MID,
    DST,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)
print("Done!")

import os
print(f"\nFinal sizes:")
print(f"  {SRC}  : {round(os.path.getsize(SRC)/1024/1024,1)} MB")
print(f"  {MID}  : {round(os.path.getsize(MID)/1024/1024,1)} MB")
print(f"  {DST}  : {round(os.path.getsize(DST)/1024/1024,1)} MB")
