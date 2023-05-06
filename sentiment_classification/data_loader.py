
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


MAX_LEN = 160
BATCH_SIZE = 16


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len, include_raw_text=False):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_raw_text = include_raw_text

    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            review, 
            add_special_tokens = True, 
            max_length = self.max_len, 
            return_token_type_ids = False, 
            return_attention_mask = True, 
            truncation = True,
            pad_to_max_length = True, 
            return_tensors = 'pt',)

        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
        if self.include_raw_text:
            output['review_text'] = review
            
        return output 


def create_data_loader(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, include_raw_text=False):
    ds = GPReviewDataset(
        reviews = df.content.to_list(), 
        targets = df.sentiment.to_list(), 
        tokenizer = tokenizer, 
        max_len = max_len,
        include_raw_text = include_raw_text)
    return DataLoader(ds, batch_size=batch_size)

def map_sentiment_scores(score_value):
    score_value = int(score_value)
    if score_value<=2:
        return 0
    elif score_value == 3:
        return 1
    else:
        return 2
    

df = pd.read_csv("data/reviews.csv")
df['sentiment'] = df.score.apply(map_sentiment_scores)
class_names = ['negative', 'neutral', 'positive']


df_train, df_test = train_test_split(df, test_size = 0.1, random_state = RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size = 0.5, random_state = RANDOM_SEED)
print(df_train.shape, df_val.shape, df_test.shape)