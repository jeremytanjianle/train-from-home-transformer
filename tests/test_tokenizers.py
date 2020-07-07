from random import randint, shuffle
from random import random as rand

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from tensorboardX import SummaryWriter
import os
import src.model.models as models
from tqdm import tqdm
import src.model.optim as optim
import src.model.train as train
from torch.utils.data import Dataset, DataLoader

from src.utils import set_seeds, get_device, truncate_tokens_pair, _sample_mask
import src.data.tokenization as tokenization
from src.data.data import Preprocess4Pretrain, SentPairDataset, seq_collate


def newyork_test_tokenizer(tokenizer):
    string = "New York"

    tokenized_string = tokenizer.tokenize(string) 
    print(f"tokenized string: {tokenized_string}")

    tokenized_string_id = tokenizer.convert_tokens_to_ids(tokenized_string)
    print(f"tokenized string id: {tokenized_string_id}")

    recovered_pieces =  tokenizer.convert_ids_to_tokens(tokenized_string_id)
    print(f"recovered pieces: {recovered_pieces}")

    assert len(tokenized_string) == len(tokenized_string_id)
    assert len(tokenized_string_id) == len(recovered_pieces)
    assert tokenized_string == recovered_pieces

    return True

def test_sentencepiece_tokenizer():
    newyork_test_tokenizer( tokenization.SPTokenizer('./data/spm.model') )

def test_unigram_tokenizer():
    newyork_test_tokenizer( tokenization.FullTokenizer(vocab_file='./data/vocab.txt', do_lower_case=True) )