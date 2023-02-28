import sys
from transformers import AutoImageProcessor, ViTModel
import torch
from datasets import load_dataset

to_fuse = (len(sys.argv) > 1 and sys.argv[1] == '1')

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

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

model.eval()

inputs = image_processor(image, return_tensors="pt")

# torch.onnx.export(model, inputs["pixel_values"], 'vit_opt.onnx',
#     input_names=["input"], output_names=["output"],
#     dynamic_axes={'input': {0:'batch'}, 'output': {0:'batch'}})

if to_fuse:
    import vit_optimizer
    vit_optimizer.optimize_bert_encoder(model)

# warm up
for _ in range(10):
    model(**inputs)

# benchmark
import time
start = time.time()
for _ in range(1000):
    model(**inputs)

# profile(model, inputs)
end = time.time()
print('1000 times inference latency: %f seconds' % (end - start))