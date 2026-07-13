import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

model_input = "model.onnx"
model_output = "model_quant.onnx"

print(f"Quantizing {model_input} to {model_output}...")
quantize_dynamic(
    model_input,
    model_output,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"]
)
print("Done!")
