import time
import json
import torch
import argparse

def profile(model, inp):
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=-1))
        prof.export_chrome_trace(f"profile_json/{prof.step_num}.json")
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=3),
            # on_trace_ready=trace_handler,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    ) as p:
        for _ in range(20):
            pred = model(*inp)
            p.step()
        return pred

def generate_model(name, default_path_prefix="models/"):
    model_file = default_path_prefix + name + ".pt"
    return torch.jit.load(model_file, map_location=torch.device('cpu'))

def generate_inputs(name, default_path_prefix="models/", batch_size=1):
    _inputs = []
    json_file = default_path_prefix + name + ".json"
    with open(json_file, 'r') as js:
        text = json.loads("{" + js.read() + "}")
        for item in text["graph_inputs"]:
            _dtype = item["dtype"]
            if "int" in str(_dtype):
                _input = torch.randint(0, 5, [batch_size + item["shape"][1:]], dtype=getattr(torch, _dtype))
            else:
                _input = torch.rand(batch_size, *item["shape"][1:], dtype=getattr(torch, _dtype))
            _inputs.append(_input)
    return _inputs

def main(opt):
    # torch.set_num_threads(opt.torch_th)
    # default_models = ["product", "video_clip", "yuanjun", "yueyi_cv", "zk"]
    # default_models = ["video_clip", "yueyi_cv"]
    default_models = ["yueyi_cv"]
    models = opt.models if len(opt.models) > 0 else default_models
    for name in models:
        model = generate_model(name)
        inputs = generate_inputs(name, batch_size=opt.batch_size)
        # model.eval()
        # torch.onnx.export(model, inputs, name + ".onnx")
        import onnxruntime as ort
        import numpy as np
        # x, y = test_data[0][0], test_data[0][1]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        model = ort.InferenceSession('models/yueyi_cv.onnx', sess_options=sess_options)
        # model = ort.InferenceSession('models/yueyi_cv_bf16.onnx', sess_options=sess_options)
        # _input = inputs[0].numpy().astype(np.bfloat16)
        _input = inputs[0].numpy()
        outputs = model.run(None, {'modelInput': _input})
        # print(outputs)

        # warmup
        for i in range(3):
            _ = model.run(None, {'modelInput': _input})

        tic = time.time()
        for _ in range(opt.iters):
            _ = model.run(None, {'modelInput': _input})
        costTime = time.time()-tic  # 总耗时
        print()
        print(">" * 15, "test", ">" * 15)
        print('>>> ', name, ': total time ', costTime, 's, qps:', opt.iters / costTime)
        print("<" * 15, "test", "<" * 15)
        print()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='', help='models, e.g. zk,yuanjun,...')
    parser.add_argument('--backends', type=str, default='', help='backends, e.g. pytorch/onnx/openvino')
    parser.add_argument('--dtype', type=str, default='fp32', help='int8/bf16/fp32')
    parser.add_argument('--torch_th', type=int, default=1, help='Torch num threads.')
    parser.add_argument('--batch_size', type=int, default=1, help='Set batch size.')
    parser.add_argument('--iters', type=int, default=1000, help='benchmark iteration.')
    parser.add_argument('--ipex', action='store_true', help='enable ipex.')
    parser.add_argument('--inc', action='store_true', help='enable inc.')
    parser.add_argument('--timeline', action='store_true', help='track timeline.')
    parser.add_argument('--graph', action='store_true', help='track graph.')
    opt = parser.parse_args()
    opt.models = str(opt.models).split(",") if len(opt.models) > 0 else []
    if opt.ipex:
        import intel_extension_for_pytorch as ipex
    main(opt=opt)