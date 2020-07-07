"""
Train sentencepiece model

Sample usage:
>>> python train_sentencepiece -i "data/wiki.train.tokens" -v 15000
"""

import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default='data/wiki.train.tokens', help="path of input corpus")
parser.add_argument("-v", "--vocab_size", default='15000', help="vocab size")
args = parser.parse_args()

params = (f'--input={args.input} --model_prefix=data/spm --vocab_size={args.vocab_size}')
spm.SentencePieceTrainer.train(params)