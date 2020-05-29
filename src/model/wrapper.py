from random import randint, shuffle
from random import random as rand

import os
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from config import CONFIG
import src.data.tokenization as tokenization
import src.model.models as models
import src.model.optim as optim
import src.model.train as train
from src.utils import set_seeds, get_device
from src.data.data import SentPairDataset, Pipeline, Preprocess4Pretrain, seq_collate


class Generator(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.hidden, 2)

        # decoder is shared with embedding layer
        ## project hidden layer to embedding layer
        embed_weight2 = self.transformer.embed.tok_embed2.weight
        n_hidden, n_embedding = embed_weight2.size()
        self.decoder1 = nn.Linear(n_hidden, n_embedding, bias=False)
        self.decoder1.weight.data = embed_weight2.data.t()

        ## project embedding layer to vocabulary layer
        embed_weight1 = self.transformer.embed.tok_embed1.weight
        n_vocab, n_embedding = embed_weight1.size()
        self.decoder2 = nn.Linear(n_embedding, n_vocab, bias=False)
        self.decoder2.weight = embed_weight1

        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder2(self.decoder1(h_masked)) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf


class Discriminator(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.hidden, 2)

        # decoder is shared with embedding layer
        ## project hidden layer to embedding layer
        self.discriminator = nn.Linear(cfg.hidden, 1, bias=False)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        cls_h = self.activ1(self.fc(h[:, 0]))

        logits = self.discriminator(h)
        logits_clsf = self.classifier(cls_h)

        return logits, logits_clsf


class ELECTRA():

    def __init__(self, args):
        self.args = args
        set_seeds(self.args.seed)

        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab, do_lower_case=True)
        tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

        pipeline = [Preprocess4Pretrain(args.max_pred,
                                        args.mask_prob,
                                        list(tokenizer.vocab.keys()),
                                        tokenizer.convert_tokens_to_ids,
                                        self.args.max_len,
                                        args.mask_alpha,
                                        args.mask_beta,
                                        args.max_gram)]
        data_iter = DataLoader(SentPairDataset(args.data_file,
                                    self.args.batch_size,
                                    tokenize,
                                    self.args.max_len,
                                    pipeline=pipeline), 
                                batch_size=self.args.batch_size, 
                                collate_fn=seq_collate,
                                num_workers=mp.cpu_count())

        discriminator = Discriminator(self.args)
        generator = Generator(self.args)

        self.optimizer = optim.optim4GPU(self.args, generator, discriminator)
        # self.g_optimizer = optim.optim4GPU(self.args, generator)
        self.trainer = train.AdversarialTrainer(self.args, 
            discriminator, generator, 
            data_iter, 
            self.optimizer, args.ratio, args.save_dir, get_device())
        os.makedirs(os.path.join(args.log_dir, args.name), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.name)) # for tensorboardX

    def train(self):
        self.trainer.train(self.writer, model_file=None, data_parallel=False)
