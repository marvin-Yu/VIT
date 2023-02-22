import torch
import numpy as np
import openvino.runtime as ov

batch_size = 1

def cover_2_onnx(model_name, default_path_prefix="models/"):
    # load TorchScript models
    model_file = default_path_prefix + model_name + ".pt"
    jit_model = torch.jit.load(model_file, map_location=torch.device('cpu')).eval()

    # generate fake inputs
    inputs = torch.rand(batch_size, 3, 224, 224, dtype=torch.float32)

    # cover to onnx model
    torch.onnx.export(jit_model, inputs, default_path_prefix + model_name + ".onnx",
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                    'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')
    print(" ")

def exec_with_openvino(model_name, default_path_prefix="models/"):
    core = ov.Core()
    compiled_model = core.compile_model(default_path_prefix + model_name + ".onnx", "AUTO")
    infer_request = compiled_model.create_infer_request()

    # Create tensor from external memory
    memory=np.random.random((batch_size, 3, 224, 224)).astype(np.float32)
    input_tensor = ov.Tensor(array=memory)

    # Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor)
    infer_request.start_async()
    infer_request.wait()

    # Get output tensor for model with one output
    output = infer_request.get_output_tensor()
    output_buffer = output.data
    print(output_buffer)
    # output_buffer[] - accessing output tensor data

cover_2_onnx("yueyi_cv")
exec_with_openvino("yueyi_cv")