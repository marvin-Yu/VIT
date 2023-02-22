from transformers import AutoImageProcessor, ViTModel
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

model.eval()

import vit_optimizer
vit_optimizer.optimize_bert_encoder(model)
print('*** Use the FUSED BERT, INT8=%r ***' % False)

inputs = image_processor(image, return_tensors="pt")

# torch.onnx.export(model, inputs["pixel_values"], 'vit.onnx',
#     input_names=["input"], output_names=["output"],
#     dynamic_axes={'input': {0:'batch'}, 'output': {0:'batch'}})

# outputs = model(**inputs)

# warm up
for _ in range(10):
    model(**inputs)

# benchmark
import time
start = time.time()
for _ in range(10):
    model(**inputs)
end = time.time()
print('10 times inference latency: %f seconds' % (end - start))