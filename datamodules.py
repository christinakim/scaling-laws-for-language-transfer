import glob
import io
import itertools
import json
import os
import os.path
from os import path
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url
from torchtext.utils import extract_archive
from torchtext.vocab import build_vocab_from_iterator
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from transformers import GPT2Tokenizer

from data_utils import WebTextIter


def build_vocab_from_file(vocab_file):
    symbols = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            symb = line.strip().split()[0]
            symbols.append(symb)
    return {s: i for i, s in enumerate(symbols)}


def build_vocab_from_json(vocab_file):
    with open(vocab_file) as json_file:
        data = json.load(json_file)
    return dict(data)


class LMDataset(Dataset):
    def __init__(self, source, seq_len, vocab):
        self.vocab = vocab
        self.seq_len = seq_len
        self.source = source

    def get_batch(self, i):
        seq_len = min(self.seq_len, len(self.source) - 1 - i)
        data = self.source[i : i + seq_len]
        target = self.source[i + 1 : i + 1 + seq_len].reshape(-1)
        return data, target, seq_len

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index=0):
        return self.get_batch(index)


class OpenWebText2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        sequence_length: int,
        batch_size: int,
        eval_batch_size: int = None,
        data_dir="/datadrive/openwebtext2",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else 5
        self.sequence_length = sequence_length
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None):
        files = glob.glob(os.path.join(self.data_dir + "/shards", "*"))
        # files = glob.glob(os.path.join(self.data_dir, "*.jsonl.zst"))
        print(files)
        self.train_paths = files[:80]
        self.val_paths = files[80:90]
        self.test_paths = files[90:]

        vocab = build_vocab_from_json(self.data_dir + "/gpt2-vocab.json")
        self.vocab = vocab
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.train_paths = [
        #     path
        #     for idx, path in enumerate(all_paths)
        #     if idx % 10 in (0, 2, 4, 6, 8,)
        # ]
        # self.valid_paths = [
        #     path for idx, path in enumerate(all_paths) if idx % 10 in (1, 9)
        # ]
        # self.test_paths = [
        #     path for idx, path in enumerate(all_paths) if idx % 10 in (3, 7)
        # ]

    def train_dataloader(self):
        train_dataset = WebTextIter(
            dataset_paths=self.train_paths,
            seq_len=self.sequence_length,
            batch_size=self.batch_size,
            tokenizer = self.tokenizer

        )
        data_loader = DataLoader(train_dataset, batch_size=None, sampler=None)
        return data_loader

    def val_dataloader(self):
        val_dataset = WebTextIter(
            dataset_paths=self.val_paths,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
            tokenizer = self.tokenizer
        )

        data_loader = DataLoader(val_dataset, batch_size=None, sampler=None,)
        return data_loader

    def test_dataloader(self):
        test_dataset = WebTextIter(
            dataset_paths=self.test_paths,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
                        tokenizer = self.tokenizer

        )
        return DataLoader(test_dataset, batch_size=None, sampler=None)


class FileDataModule(OpenWebText2DataModule):
    def __init__(
        self,
        sequence_length: int,
        batch_size: int,
        eval_batch_size: int = None,
        data_dir="/datadrive/",
    ):
        super().__init__(sequence_length, batch_size, eval_batch_size, data_dir)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else 5
        self.sequence_length = sequence_length
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None):
        self.train_paths = [self.data_dir + "/train.txt"]
        self.val_paths = [self.data_dir + "/valid.txt"]
        self.test_paths = [self.data_dir + "/test.txt"]

        self.all_paths = self.train_paths + self.val_paths + self.test_paths
        if not os.path.exists(self.data_dir+ "/tokenizer.json"):

            tokenizer = Tokenizer(BPE())

            tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer()

            tokenizer.train(files=self.all_paths, trainer=trainer)
            #special_tokens_dict = ['bos_token','eos_token']
            special_tokens_dict = ['<bos>', '<eos>']
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            tokenizer = tokenizer.save(self.data_dir + "/tokenizer.json")

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.data_dir + "/tokenizer.json", unk_token='<unk>', bos_token='<bos>', eos_token='<eos>')

        self.vocab = self.tokenizer.get_vocab()

        # self.train_paths = [
        #     path
        #     for idx, path in enumerate(all_paths)
        #     if idx % 10 in (0, 2, 4, 6, 8,)
        # ]
        # self.valid_paths = [
        #     path for idx, path in enumerate(all_paths) if idx % 10 in (1, 9)
        # ]
        # self.test_paths = [
        #     path for idx, path in enumerate(all_paths) if idx % 10 in (3, 7)
        # ]



class WikiText2DataModule(pl.LightningDataModule):
    def __init__(
        self, sequence_length: int, batch_size: int, eval_batch_size: int = None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.sequence_length = sequence_length

    def data_process(self, raw_text_iter, tokenizer, vocab):
        data = [
            torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
            for item in raw_text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, bsz):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    def setup(self, stage: Optional[str] = None):
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
        test_filepath, valid_filepath, train_filepath = extract_archive(
            download_from_url(url)
        )
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(
            map(tokenizer, iter(io.open(train_filepath, encoding="utf8")))
        )
        self.vocab = vocab

        self.train_data = self.data_process(
            iter(io.open(train_filepath, encoding="utf8")), tokenizer, vocab
        )
        self.val_data = self.data_process(
            iter(io.open(valid_filepath, encoding="utf8")), tokenizer, vocab
        )
        self.test_data = self.data_process(
            iter(io.open(test_filepath, encoding="utf8")), tokenizer, vocab
        )

        if stage == "fit" or stage is None:
            self.train_data = self.batchify(self.train_data, self.batch_size)
            self.val_data = self.batchify(self.val_data, self.eval_batch_size)
        if stage == "test" or stage is None:
            self.test_data = self.batchify(self.test_data, self.eval_batch_size)

    def train_dataloader(self):

        data = self.train_data
        train_dataset = LMDataset(data, self.sequence_length, self.batch_size)
        data_loader = DataLoader(train_dataset, batch_size=None)
        return data_loader

    def val_dataloader(self):
        data = self.val_data
        val_dataset = LMDataset(data, self.sequence_length, self.batch_size)

        return DataLoader(val_dataset, batch_size=None)

    def test_dataloader(self):
        if self.trainer.on_gpu and self.trainer.gpus > 1:
            data = [
                x
                for i, x in enumerate(self.test_data)
                if i % self.trainer.gpus == self.trainer.global_rank
            ]
        else:
            data = self.test_data
        return DataLoader(data, num_workers=24, batch_size=None)

