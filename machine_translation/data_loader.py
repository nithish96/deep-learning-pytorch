from io import open
import unicodedata
import string
import re
import random
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from collections import Counter 

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

def unicodeToAscii(s):
    return ''.join(
      c for c in unicodedata.normalize('NFD', s) 
      if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z.!?]+", " ", s)
    return s

def filterPair(p, max_length, prefixes):
    good_length = (len(p[0].split(' ')) < max_length) and (len(p[1].split(' ')) < max_length)
    if len(prefixes) == 0:
        return good_length
    else:
        return good_length and p[0].startswith(prefixes)

def filterPairs(pairs, max_length, prefixes=()):
    return [pair for pair in pairs if filterPair(pair, max_length, prefixes)]
     

def prepareData(lines, filter=False, reverse=False, max_length=10, prefixes=()):
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    print(f"Given {len(pairs):,} sentence pairs.")
    if filter:
        pairs = filterPairs(pairs, max_length=max_length, prefixes=prefixes)
        print(f"After filtering, {len(pairs):,} remain.")

    return pairs

def get_pairs():

    with open('data/eng-fra.txt', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    pairs = prepareData(lines, filter=True, 
                        max_length=MAX_SENTENCE_LENGTH, 
                        prefixes=basic_prefixes if FILTER_TO_BASIC_PREFIXES else ())

    return pairs

def get_vocab(pairs, SPECIALS):
    en_list = []
    fr_list = []
    en_counter = Counter()
    fr_counter = Counter()
    en_lengths = []
    fr_lengths = []
    for en, fr in pairs:
        en_toks = en_tokenizer(en)
        fr_toks = fr_tokenizer(fr)
        en_list += [en_toks]
        fr_list += [fr_toks]
        en_counter.update(en_toks)
        fr_counter.update(fr_toks)
        en_lengths.append(len(en_toks))
        fr_lengths.append(len(fr_toks))

    en_vocab = build_vocab_from_iterator(en_list, specials=SPECIALS)
    fr_vocab = build_vocab_from_iterator(fr_list, specials=SPECIALS)

    return en_vocab, fr_vocab


def get_train_test_valid(pairs):

    VALID_PCT = 0.1
    TEST_PCT  = 0.1

    train_data = []
    valid_data = []
    test_data = []

    random.seed(6547)
    for (en, fr) in pairs:
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(en)])
        fr_tensor_ = torch.tensor([fr_vocab[token] for token in fr_tokenizer(fr)])
        random_draw = random.random()
        if random_draw <= VALID_PCT:
            valid_data.append((en_tensor_, fr_tensor_))
        elif random_draw <= VALID_PCT + TEST_PCT:
            test_data.append((en_tensor_, fr_tensor_))
        else:
            train_data.append((en_tensor_, fr_tensor_))

    return train_data, valid_data, test_data



def generate_batch(data_batch):
    en_batch, fr_batch = [], []
    for (en_item, fr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))

    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=False)
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX, batch_first=False)

    return en_batch, fr_batch


MAX_SENTENCE_LENGTH = 20
FILTER_TO_BASIC_PREFIXES = False


basic_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ",
    'are you', 'am i ', 
    'were you', 'was i ', 
    'where are', 'where is',
    'what is', 'what are'
)


pairs = get_pairs()
fr_tokenizer = get_tokenizer('spacy', language='fr')
en_tokenizer = get_tokenizer('spacy', language='en')
     
SPECIALS = ['', '<pad>', '<bos>', '<eos>']
en_vocab, fr_vocab = get_vocab(pairs, SPECIALS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, valid_data, test_data= get_train_test_valid(pairs)

print(f" Train:{len(train_data):,}. Valid: {len(valid_data):,}, Test: {len(test_data):,}""")
     
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

for en_id, fr_id in zip(en_vocab.lookup_indices(SPECIALS), fr_vocab.lookup_indices(SPECIALS)):
  assert en_id == fr_id



BATCH_SIZE = 64
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
     

for i, (en_id, fr_id) in enumerate(test_iter):
    print('English:', ' '.join([en_vocab.lookup_token(idx) for idx in en_id[:, 0]]))
    print('French:', ' '.join([fr_vocab.lookup_token(idx) for idx in fr_id[:, 0]]))
    if i == 4: 
        break
    else:
        print()