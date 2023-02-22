import time
import numpy as np
import openvino.runtime as ov

core = ov.Core()
BS = 4
compiled_model = core.compile_model("models/yueyi_cv_bf16.onnx", "AUTO")
# compiled_model = core.compile_model("models/yueyi_cv.onnx", "AUTO")
infer_request = compiled_model.create_infer_request()

# Create tensor from external memory
memory=np.random.random((BS, 3, 224, 224)).astype(np.float32)
input_tensor = ov.Tensor(array=memory, shape=[BS, 3, 224, 224], type=ov.Type.bf16)
# input_tensor = ov.Tensor(array=memory, shape=[BS, 3, 224, 224])

for i in range(3):
    # Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor)
    infer_request.start_async()
    infer_request.wait()

tic = time.time()
for _ in range(1000):
    # Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor)
    infer_request.start_async()
    infer_request.wait()
    output = infer_request.get_output_tensor()

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