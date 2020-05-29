from random import randint, shuffle
from random import random as rand

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm

import src.data.tokenization as tokenization
import src.model.models as models
import src.model.optim as optim
import src.model.train as train
from src.utils import set_seeds, get_device, truncate_tokens_pair, _sample_mask

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

def bufcount(filename):
    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines

class SentPairDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore') # for a positive sample
        self.tokenize = tokenize # tokenize function
        self.size = bufcount(file)
        self.max_len = max_len # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """ Read tokens from file pointer with limited length """
        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens # return last tokens in the document
            tokens.extend(self.tokenize(line.strip()))
        return tokens

    def __getitem__(self, idx): 
        """
        iterator to load data
        text file -> is_next, tokens_a, tokens_b -> instance
        """
        # sampling length of each tokens_a and tokens_b
        # sometimes sample a short sentence to match between train and test sequences
        # ALBERT is same  randomly generate input
        # sequences shorter than 512 with a probability of 10%.
        len_tokens = randint(1, int(self.max_len / 2)) \
            if rand() < self.short_sampling_prob \
            else int(self.max_len / 2)

        is_next = rand() < 0.5 # whether token_b is next to token_a or not

        tokens_a = self.read_tokens(self.f_pos, len_tokens, True)
        f_next = self.f_pos # `f_next` should be next point
        tokens_b = self.read_tokens(f_next, len_tokens, False)

        # there are no more tokens in the document
        # split between tokens_a and tokens_b, halfway
        if (len(tokens_b) == 0) & (len(tokens_a)>10):
            half_split = int(len(tokens_a)/2)
            tokens_b = tokens_a[half_split:]
            tokens_a = tokens_a[:half_split]

        if tokens_a is None or tokens_b is None: # end of file
            self.f_pos.seek(0, 0) # reset file pointer

            # re-read token
            tokens_a = self.read_tokens(self.f_pos, len_tokens, True)
            f_next = self.f_pos # `f_next` should be next point
            tokens_b = self.read_tokens(f_next, len_tokens, False)

        # SOP, sentence-order prediction
        instance = (is_next, tokens_a, tokens_b) if is_next \
            else (is_next, tokens_b, tokens_a)

        for proc in self.pipeline:
            instance = proc(instance)

        return instance
    
    def __len__(self):
        return self.size

def seq_collate(batch):
    # batch_tensors = [print(len(x)) for x in batch if len(x)!=400]
    batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
    return batch_tensors    


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len,
                 mask_alpha, mask_beta, max_gram):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (sub)words

        self.indexer = indexer # function from token to token index
        self.max_len = max_len
        self.mask_alpha = mask_alpha
        self.mask_beta = mask_beta
        self.max_gram = max_gram

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))

        original_ids = self.indexer(tokens)
        # For masked Language Models
        masked_tokens, masked_pos, tokens = _sample_mask(tokens, self.mask_alpha,
                                                        self.mask_beta, self.max_gram,
                                                        goal_num_predict=n_pred)
        # prev_masked_tokens = masked_tokens.copy()
        # print(len(masked_tokens))

        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        original_ids.extend([0]*n_pad)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero Padding for masked target
        # originally the author constrained masked_ids, masked_pos and masked_weights to be length 75
        # but they should be of the same length
        # so I replace max_pred w max_len
        # prev_masked_id = masked_ids.copy()
        if self.max_len > len(masked_ids):
            masked_ids.extend([0] * (self.max_len - len(masked_ids)))
        elif self.max_len > len(masked_ids):
            raise ValueError("Strangely, the masked_ids is more than max_pred")
        # assert self.max_pred == len(masked_ids), f"self.max_pred {self.max_pred} vs shape of prev {len(prev_masked_id)} vs shape of new {len(masked_ids)}"
        if self.max_len > len(masked_pos):
            masked_pos.extend([0] * (self.max_len - len(masked_pos)))
        if self.max_len > len(masked_weights):
            masked_weights.extend([0] * (self.max_len - len(masked_weights)))
        
        
        # Author implementation isn't exact the same as original bert model 
        # as masked_ids only contain the un-masked token
        # for k,v in {"input_ids":input_ids, "segment_ids":segment_ids, "input_mask":input_mask, "masked_ids":masked_ids, "masked_pos":masked_pos, "masked_weights":masked_weights, "is_next":is_next, "original_ids":original_ids}.items():
        #     assert len(v) == self.max_len, f"unexpected shape in {k}: {v}"
        # print(f"Shape of masked_ids: {len(masked_ids)}")
        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, original_ids)


if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer(vocab_file='./data/vocab.txt', do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(75,
                                    0.15,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    400,
                                    1,
                                    1,
                                    3)]
    data_iter = DataLoader(SentPairDataset('./data/wiki.test.tokens',
                                16,
                                tokenize,
                                400,
                                pipeline=pipeline), batch_size=16, collate_fn=seq_collate, num_workers=8)

    for batch in tqdm(data_iter):
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, original_ids = batch
