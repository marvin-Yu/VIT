import time
import numpy as np
import openvino.runtime as ov
import sys
from transformers import AutoImageProcessor, ViTModel
import torch
from datasets import load_dataset

batch_size = 1

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image = [image for _ in range(batch_size)]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

inputs = image_processor(image, return_tensors="pt")

config={}
config['INFER_PRECISION'] = 'f32'

core = ov.Core()
core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "f32"})

# compiled_model = core.compile_model(model, "CPU", config)
compiled_model = core.compile_model("vit.onnx", "AUTO")
# compiled_model = core.compile_model("models/yueyi_cv.onnx", "AUTO")
# infer_request = compiled_model.create_infer_request()

# Create tensor from external memory
# memory=np.random.random((batch_size, 3, 224, 224)).astype(np.float32)
memory=inputs["pixel_values"].numpy()
# input_tensor = ov.Tensor(array=memory, shape=[batch_size, 3, 224, 224], type=ov.Type.bf16)
input_tensor = ov.Tensor(array=memory, shape=[batch_size, 3, 224, 224])

# print(compiled_model(input_tensor)[compiled_model.output(0)])

for i in range(10):
    # Set input tensor for model with one input
    compiled_model(input_tensor)[compiled_model.output(0)]
    pass

tic = time.time()
for _ in range(1000):
    # Set input tensor for model with one input
    compiled_model(input_tensor)

costTime = time.time()-tic  # 总耗时
print()
print(">" * 15, "test", ">" * 15)
print('>>> ', "yueyi_cv", ': total time ', costTime, 's, qps:', 1000 / costTime)
print("<" * 15, "test", "<" * 15)
print()

# # Set input tensor for model with one input
# infer_request.set_input_tensor(input_tensor)
# infer_request.start_async()
# infer_request.wait()

# Get output tensor for model with one output
# output = infer_request.get_output_tensor()
# output_buffer = output.data
# print(output_buffer)
# output_buffer[] - accessing output tensor data

# print(compiled_model(input_tensor)[compiled_model.output(0)])
