CONFIG_DICT={
    # general config
    "data_file": "./data/wiki.train.tokens", 
    "generator_cfg": "./config/generator_base.json", 
    "log_dir": "./logs", 
    "mask_alpha": 4, 
    "mask_beta": 1, 
    "mask_prob": 0.15, 
    "max_gram": 3, 
    "max_pred": 75, 

    # model config
    "mode": "electra", 
    "embedding" : 64,
    "hidden": 256,
    "hidden_ff": 1024,
    "n_layers": 12,
    "n_heads": 4,
    "max_len": 75,
    "p_drop_hidden": 0.1,
    "n_segments": 2,

    # tokenizer config
    "tokenizer": "sentencepiece", # "bert_unigram"
    "model_path": "./data/spm.model",
    "sp_nbest_size":-1,
    "sp_alpha": 0.1,
    "vocab": "./data/spm.vocab", # "./data/vocab.txt",
    "vocab_size": 15000, # 30522

    # training config
    "name": "electra", 
    "ratio": 50, 
    "save_dir": "./saved", 
    "seed": 128,
    "batch_size": 5,
    "lr": 5e-4,
    "n_epochs": 10,
    "warmup": 0.1,
    "save_steps": 10000,
    "total_steps": 1000000,

    # downstream
    "pretrain":"./saved/discriminator.pt",
    "task":"mrpc", # sst
    "train":"./data/MRPC/train.tsv",
    "eval":"./data/MRPC/dev.tsv",
    "downstream_batch_size": 18,
    "downstream_lr": 2e-4,
    "downstream_n_epochs": 4,
    "downstream_warmup": 0.1,
    "downstream_save_steps": 300, # 1000
    "downstream_total_steps": 800, # 11000
}


class config:
    def __init__(self, CONFIG_DICT=CONFIG_DICT):
        self.CONFIG_DICT = CONFIG_DICT
        for k, v in CONFIG_DICT.items():
            setattr(self, k, v)

CONFIG=config()
            