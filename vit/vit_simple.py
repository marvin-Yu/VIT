import sys
from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset

to_fuse = (len(sys.argv) > 1 and sys.argv[1] == '1')

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

model.eval()

inputs = image_processor(image, return_tensors="pt")

if to_fuse:
    import vit_optimizer
    vit_optimizer.optimize_bert_encoder(model)

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

print(model(**inputs))