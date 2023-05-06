from datasets import load_dataset

from pathlib import Path
import numpy as np
import torch 
from transformers import * 

device = "cuda" if torch.cuda.is_available() else "cpu"

target_path = '/data/nithish/others/deep-learning-pytorch/image_captioning/data/'

max_length = 32 # max length of the captions in tokens
coco_dataset_ratio = 50 # 50% of the COCO2014 dataset
train_ds = load_dataset("HuggingFaceM4/COCO", split=f"train[:{coco_dataset_ratio}%]", cache_dir=target_path)
valid_ds = load_dataset("HuggingFaceM4/COCO", split=f"validation[:{coco_dataset_ratio}%]", cache_dir=target_path)
test_ds = load_dataset("HuggingFaceM4/COCO", split="test", cache_dir=target_path)
print(len(train_ds), len(valid_ds), len(test_ds))

# train_ds = train_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)
# valid_ds = valid_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)
# test_ds = test_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)

# def preprocess(items):
#   pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
#   sentences = [ sentence["raw"] for sentence in items["sentences"] ]
#   targets = tokenizer(sentences, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
#   return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}

# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#         'labels': torch.stack([x['labels'] for x in batch])
#     }

# train_dataset = train_ds.with_transform(preprocess)
# valid_dataset = valid_ds.with_transform(preprocess)
# test_dataset  = test_ds.with_transform(preprocess)