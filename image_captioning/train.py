
import torch
from PIL import Image


import transformers
transformers.logging.set_verbosity_error()


from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import * 

from tqdm import tqdm
import evaluate

from datasets import load_dataset

from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder_model = "WinKawaks/vit-small-patch16-224"
decoder_model = "prajjwal1/bert-tiny"
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model, decoder_model
).to(device)

tokenizer = AutoTokenizer.from_pretrained(decoder_model)
image_processor = ViTImageProcessor.from_pretrained(encoder_model)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id


target_path = '/data/nithish/others/deep-learning-pytorch/image_captioning/data/'

max_length = 32 # max length of the captions in tokens
coco_dataset_ratio = 50 # 50% of the COCO2014 dataset
train_ds = load_dataset("HuggingFaceM4/COCO", split=f"train[:{coco_dataset_ratio}%]", cache_dir=target_path)
valid_ds = load_dataset("HuggingFaceM4/COCO", split=f"validation[:{coco_dataset_ratio}%]", cache_dir=target_path)
test_ds = load_dataset("HuggingFaceM4/COCO", split="test", cache_dir=target_path)
print(len(train_ds), len(valid_ds), len(test_ds))


train_ds = train_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=5)
valid_ds = valid_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=5)
test_ds = test_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=5)

def preprocess(items):
  pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
  sentences = [ sentence["raw"] for sentence in items["sentences"] ]
  targets = tokenizer(sentences, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
  return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }

train_dataset = train_ds.with_transform(preprocess)
valid_dataset = valid_ds.with_transform(preprocess)
test_dataset  = test_ds.with_transform(preprocess)


bleu = evaluate.load("bleu")  
def compute_metrics(eval_pred):
    preds = eval_pred.label_ids
    labels = eval_pred.predictions
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu_result = bleu.compute(predictions=pred_str, references=labels_str)
    generation_length = bleu_result["translation_length"]
    return {
        "bleu": round(bleu_result["bleu"] * 100, 4), 
        "gen_len": bleu_result["translation_length"] / len(preds)
    }

batch_size = 16 
train_dataset_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
valid_dataset_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
test_dataset_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)

num_epochs = 10 # number of epochs
batch_size = 64 # the size of batches
n_valid_steps = len(valid_dataset_loader)
current_step = 0

def get_valid_metrics():
    valid_loss = 0.0
    predictions, labels = [], []    
    for batch in valid_dataset_loader:
        pixel_values = batch["pixel_values"]
        label_ids = batch["labels"]
        outputs = model(pixel_values=pixel_values, labels=label_ids)
        loss = outputs.loss
        valid_loss += loss.item()
        logits = outputs.logits.detach().cpu()
        predictions.extend(logits.argmax(dim=-1).tolist())
        labels.extend(label_ids.tolist())
    eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
    metrics = compute_metrics(eval_prediction)
    return metrics, valid_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

best_valid_bleu=0.0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataset_loader, "Training", total=len(train_dataset_loader), leave=False):
        pixel_values, labels = batch["pixel_values"], batch["labels"]
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_v = loss.item()
        train_loss += loss_v
        current_step += 1

    model.eval()
    metrics, valid_loss = get_valid_metrics() 
    
    print(f"\nEpoch: {epoch},  Train Loss: {train_loss / len(train_dataset_loader):.4f}, " + 
        f"Valid Loss: {valid_loss / n_valid_steps:.4f}, BLEU: {metrics['bleu']:.4f}\n")
    
    if metrics['bleu'] > best_valid_bleu:
        model.save_pretrained(f"./models/best-checkpoint/")
        tokenizer.save_pretrained(f"./models/best-checkpoint/")
        image_processor.save_pretrained(f"./models/best-checkpoint/")
        best_valid_bleu = metrics['bleu']
    

