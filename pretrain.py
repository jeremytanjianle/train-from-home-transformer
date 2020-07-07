"""
Sample usage:
>>> python pretrain.py
"""
from random import randint, shuffle
from random import random as rand

import numpy as np
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter
import os
import multiprocessing as mp

from config import CONFIG
from src.model.wrapper import ELECTRA


if __name__ == '__main__':
    
    trainer = ELECTRA(args=CONFIG)
    trainer.train()