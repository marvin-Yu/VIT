import time
import json
import torch
import argparse

def generate_model(name, default_path_prefix="models/"):
    model_file = default_path_prefix + name + ".pt"
    return torch.jit.load(model_file, map_location=torch.device('cpu'))

def generate_inputs(name, default_path_prefix="models/"):
    _inputs = []
    json_file = default_path_prefix + name + ".json"
    with open(json_file, 'r') as js:
        text = json.loads("{" + js.read() + "}")
        for item in text["graph_inputs"]:
            _dtype = item["dtype"]
            if "int" in str(_dtype):
                _input = torch.randint(0, 5, item["shape"], dtype=getattr(torch, _dtype))
            else:
                _input = torch.rand(*item["shape"], dtype=getattr(torch, _dtype))
            _inputs.append(_input)
    return _inputs

def main(opt):
    # default_models = ["product", "video_clip", "yuanjun", "yueyi_cv", "zk"]
    default_models = ["video_clip", "yueyi_cv"]
    models = opt.models if len(opt.models) > 0 else default_models
    for name in models:
        model = generate_model(name)
        inputs = generate_inputs(name)

        # warmup
        for i in range(3):
            _ = model(*inputs)

        tic = time.time()
        for _ in range(opt.iters):
            _ = model(*inputs)
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
    parser.add_argument('--torch_th', type=int, default=1, help='Torch num threads.')
    parser.add_argument('--iters', type=int, default=1000, help='benchmark iteration.')
    parser.add_argument('--ipex', action='store_true', help='enable ipex.')
    parser.add_argument('--inc', action='store_true', help='enable inc.')
    opt = parser.parse_args()
    opt.models = str(opt.models).split(",") if len(opt.models) > 0 else []
    if opt.ipex:
        import intel_extension_for_pytorch as ipex
    main(opt=opt)