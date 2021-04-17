import glob
import json
import os
import os.path
from typing import Optional

import pytorch_lightning as pl
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from transformers import PreTrainedTokenizerFast

from data_utils import ChineseWebtextDataset
from data_utils import OscarDataset
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
        self.train_paths = files[:80]
        self.val_paths = files[80:90]
        self.test_paths = files[90:]

        vocab = build_vocab_from_json(self.data_dir + "/gpt2-vocab.json")
        self.vocab = vocab
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def train_dataloader(self):
        train_dataset = WebTextIter(
            dataset_paths=self.train_paths,
            seq_len=self.sequence_length,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
        )
        data_loader = DataLoader(train_dataset, batch_size=None, sampler=None)
        return data_loader

    def val_dataloader(self):
        val_dataset = WebTextIter(
            dataset_paths=self.val_paths,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
            tokenizer=self.tokenizer,
        )

        data_loader = DataLoader(val_dataset, batch_size=None, sampler=None,)
        return data_loader

    def test_dataloader(self):
        test_dataset = WebTextIter(
            dataset_paths=self.test_paths,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
            tokenizer=self.tokenizer,
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
        if not os.path.exists(self.data_dir + "/tokenizer.json"):

            tokenizer = Tokenizer(BPE())

            tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer()

            tokenizer.train(files=self.all_paths, trainer=trainer)
            # special_tokens_dict = ['bos_token','eos_token']
            special_tokens_dict = ["<bos>", "<eos>"]
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print("We have added", num_added_toks, "tokens")
            tokenizer = tokenizer.save(self.data_dir + "/tokenizer.json")

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.data_dir + "/tokenizer.json",
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
        )

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


class ChineseWebtextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sequence_length: int,
        batch_size: int,
        token_limit: int,
        eval_batch_size: int = None,
        data_dir="/datadrive/",
        diff_tokenization=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else 5
        self.sequence_length = sequence_length
        self.data_dir = data_dir
        self.token_limit = token_limit
        if diff_tokenization:
            print("diff tokenizer")

            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=self.data_dir + "/tokenizer.json",
                unk_token="<unk>",
                bos_token="<bos>",
                eos_token="<eos>",
            )
        else:
            print("same tokenizer")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.vocab = self.tokenizer.get_vocab()

    def setup(self, stage: Optional[str] = None):
        self.test_file = self.data_dir + "/web_text_zh_testa.json"
        self.train_file = self.data_dir + "/web_text_zh_train.json"
        self.valid_file = self.data_dir + "/web_text_zh_valid.json"

    def train_dataloader(self):
        train_dataset = ChineseWebtextDataset(
            file=self.train_file,
            seq_len=self.sequence_length,
            batch_size=self.batch_size,
            token_limit=self.token_limit,
            tokenizer=self.tokenizer,
        )
        data_loader = DataLoader(train_dataset, batch_size=None, sampler=None)
        return data_loader

    def val_dataloader(self):
        val_dataset = ChineseWebtextDataset(
            file=self.valid_file,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
            token_limit=self.token_limit,
            tokenizer=self.tokenizer,
        )

        data_loader = DataLoader(val_dataset, batch_size=None, sampler=None,)
        return data_loader

    def test_dataloader(self):
        test_dataset = ChineseWebtextDataset(
            file=self.test_file,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
            token_limit=self.token_limit,
            tokenizer=self.tokenizer,
        )
        return DataLoader(test_dataset, batch_size=None, sampler=None)


class OscarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sequence_length: int,
        batch_size: int,
        token_limit: int,
        eval_batch_size: int = None,
        data_dir="/datadrive/",
        diff_tokenization=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else 5
        self.sequence_length = sequence_length
        self.data_dir = data_dir
        self.token_limit = token_limit
        if diff_tokenization:
            print("diff tokenizer")

            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=self.data_dir + "/tokenizer.json",
                unk_token="<unk>",
                bos_token="<bos>",
                eos_token="<eos>",
            )
        else:
            print("same tokenizer")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.vocab = self.tokenizer.get_vocab()

    def setup(self, stage: Optional[str] = None):
        self.test_file = self.data_dir + "/test.txt"
        self.train_file = self.data_dir + "/train.txt"
        self.valid_file = self.data_dir + "/valid.txt"

    def train_dataloader(self):
        train_dataset = OscarDataset(
            file=self.train_file,
            seq_len=self.sequence_length,
            batch_size=self.batch_size,
            token_limit=self.token_limit,
            tokenizer=self.tokenizer,
        )
        data_loader = DataLoader(train_dataset, batch_size=None, sampler=None)
        return data_loader

    def val_dataloader(self):
        val_dataset = OscarDataset(
            file=self.valid_file,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
            token_limit=self.token_limit,
            tokenizer=self.tokenizer,
        )

        data_loader = DataLoader(val_dataset, batch_size=None, sampler=None,)
        return data_loader

    def test_dataloader(self):
        test_dataset = OscarDataset(
            file=self.test_file,
            seq_len=self.sequence_length,
            batch_size=self.eval_batch_size,
            token_limit=self.token_limit,
            tokenizer=self.tokenizer,
        )
        return DataLoader(test_dataset, batch_size=None, sampler=None)
