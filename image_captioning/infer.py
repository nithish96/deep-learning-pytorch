import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

import transformers
transformers.logging.set_verbosity_error()

finetuned_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
finetuned_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
finetuned_image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# finetuned_model = VisionEncoderDecoderModel.from_pretrained("./models/best-checkpoint").to(device)
# finetuned_tokenizer = GPT2TokenizerFast.from_pretrained("./models/best-checkpoint")
# finetuned_image_processor = ViTImageProcessor.from_pretrained("./models/best-checkpoint")


import urllib.parse as parse
import os

def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
        
def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    img = image_processor(image, return_tensors="pt").to(device)
    output = model.generate(**img)
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption

url = "http://images.cocodataset.org/test-stuff2017/000000009384.jpg"
caption = get_caption(finetuned_model, finetuned_image_processor, finetuned_tokenizer, url)

print(caption)
