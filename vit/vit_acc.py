import sys
from sklearn.metrics import accuracy_score
import numpy as np
import os
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
from pathlib import Path

model_id="nateraw/vit-base-beans"
model_name="vit-base-beans"

image_processor = ViTImageProcessor.from_pretrained(model_id)
fp32_model = ViTForImageClassification.from_pretrained(model_id)

# import vit_optimizer
# vit_optimizer.optimize_bert_encoder(fp32_model, True)
# int8_model = torch.quantization.quantize_dynamic(fp32_model)

eval_dataset = load_dataset("beans",split=["test"])[0]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return dict(accuracy=accuracy_score(predictions, labels))

def predict(model, image):
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        rtn = model(**inputs)
    return rtn

size = len(eval_dataset["image"])
# size = 1

int8_model = torch.quantization.quantize_dynamic(fp32_model)
# fp32_eval_pred = ([predict(fp32_model, eval_dataset["image"][i])["logits"].numpy() for i in range(size)], eval_dataset["labels"])
int8_eval_pred = ([predict(int8_model, eval_dataset["image"][i])["logits"].numpy() for i in range(size)], eval_dataset["labels"][0:size])

# fp32_accuracy = compute_metrics(fp32_eval_pred)
int8_accuracy = compute_metrics(int8_eval_pred)

# print(f"fp32_accuracy: {fp32_accuracy['accuracy']*100:.2f}%")
print(f"origin int8_accuracy: {int8_accuracy['accuracy']*100:.2f}%")
# print(f"The quantized model achieves {round(int8_accuracy['accuracy']/fp32_accuracy['accuracy'],4)*100:.2f}% accuracy of the fp32 model")

import vit_optimizer
vit_optimizer.optimize_bert_encoder(fp32_model, True)
int8_model = torch.quantization.quantize_dynamic(fp32_model)

int8_eval_pred = ([predict(int8_model, eval_dataset["image"][i])["logits"].numpy() for i in range(size)], eval_dataset["labels"][0:size])

int8_accuracy = compute_metrics(int8_eval_pred)
print(f"fused int8_accuracy: {int8_accuracy['accuracy']*100:.2f}%")