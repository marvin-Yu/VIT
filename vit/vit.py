import sys
from transformers import AutoImageProcessor, ViTModel
import torch
from datasets import load_dataset

to_fuse = (len(sys.argv) > 1 and sys.argv[1] == '1')
to_quantize = (len(sys.argv) > 2 and sys.argv[2] == '1')
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
iters = 1000

def profile(model, inp):
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range((1 + 1 + 3) * 1):
            pred = model(**inp)
            prof.step()
        return pred


dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image = [image for _ in range(batch_size)]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

model.eval()

inputs = image_processor(image, return_tensors="pt")

# torch.onnx.export(model, inputs["pixel_values"], 'vit.onnx',
#     input_names=["input"], output_names=["output"],
#     dynamic_axes={'input': {0:'batch'}, 'output': {0:'batch'}})

if to_fuse:
    import vit_optimizer
    vit_optimizer.optimize_bert_encoder(model, to_quantize)

# Dynamic quantization with PT
if to_quantize:
    print('*** PT dynamic quantization (torch.quantization.quantize_dynamic) ***')
    model = torch.quantization.quantize_dynamic(model)

# print(model(**inputs))

# warm up
for _ in range(10):
    model(**inputs)

# benchmark
import time
start = time.time()
for _ in range(iters):
    model(**inputs)

# profile(model, inputs)
end = time.time()
print(str(iters) + ' times inference latency: %f QPS' % (iters / (end - start)))