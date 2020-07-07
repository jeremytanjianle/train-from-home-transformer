"""
Train sentencepiece model

Sample usage:
>>> python train_sentencepiece -i "data/wiki.train.tokens"
"""

import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default='data/wiki.train.tokens', help="path of input corpus")
args = parser.parse_args()

params = (f'--input={args.input} --model_prefix=data/spm --vocab_size=5000')
spm.SentencePieceTrainer.train(params)