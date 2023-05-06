
import torch
import torch.nn.functional as F

from transformers import RobertaTokenizer

from model import model 


MAX_LEN = 160
class_names = ['negative', 'neutral', 'positive']

PRE_TRAINED_MODEL_NAME = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'best_model_state.pt'
model.load_state_dict(torch.load(model_path, map_location=device))

review_text = "I hate completing my todos! Worst app ever!!!"
encoded_review = tokenizer.encode_plus(review_text, max_length=MAX_LEN, add_special_tokens=True, return_token_type_ids=False, pad_to_max_length=True, return_attention_mask=True,
                                       truncation=True, return_tensors='pt')
input_ids = encoded_review['input_ids'].to(device)
attention_mask=encoded_review['attention_mask'].to(device)
output = model(input_ids, attention_mask)
_,prediction = torch.max(output, dim=1)

print(f'Review text: {review_text}')
print(f'Sentiment  : {class_names[prediction]}')

