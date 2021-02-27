import os
import glob
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from transformers import GPT2Tokenizer

from utils.vocabulary import Reader
from utils.vocabulary import Vocab
import random
from itertools import chain, cycle, islice
import torch.utils.data as data

import time
import torch
import numpy as np


class WebTextDocumentIterator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __iter__(self):
        reader = Reader()
        doc_block = 20000
        documents = []
        for i, x in enumerate(reader.read_jsonl(self.dataset_path)):
            documents.append(x)
            if len(documents) == doc_block:
                yield documents
                documents = []
        yield documents

class FileIterator:
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths

    def get_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            yield from f.readlines()

    def __iter__(self):
        for path in self.dataset_paths:
            yield from self.get_file(path)


class TokenizerIterator:
    def __init__(self, seq_len, tokenizer, seed, dataset_path):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.document_iter = WebTextDocumentIterator(dataset_path)
        self.seed = seed

    def __iter__(self):
        block = []
        for documents in self.document_iter:
            random.Random(self.seed).shuffle(documents)
            
            
            for doc_i, x in enumerate(documents):
                tokenized = self.tokenizer(text=x[1],).input_ids
                tokenized.append(self.tokenizer.eos_token_id)

                tokenized.insert(0, self.tokenizer.eos_token_id)
                tokenized_length = len(tokenized)
                for token in tokenized:
                    if len(block) == self.seq_len + 1:
                        yield block[:-1], block[1:], (x[0], doc_i)
                        block = []
                    block.append(token)

                # if len(tokenized) >= self.seq_len:
                #     i = 0
                #     while i <= len(tokenized) - self.seq_len:
                #         yield tokenized[i : i + self.seq_len]
                #         i += self.seq_len


class BatchIterator:
    def __init__(self, seq_len, batch_size, drop_last, tokenizer, dataset_paths):

        self.dataset_paths = dataset_paths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def process_data(self, seed_dataset):
        seed, dataset = seed_dataset
        self.tokenizer_iter = TokenizerIterator(self.seq_len, self.tokenizer, seed, dataset)
        for x in self.tokenizer_iter:
            yield x 
            # batch.append(x)
            # if len(batch) == self.batch_size:
            #     yield batch
            #     batch = []
            
    def shuffled_data_list(self, i):
        #split = len(self.dataset_paths) // self.batch_size
        #dataset_paths = self.dataset_paths[(i*split):((i+1)*split)]
        shuffled = self.dataset_paths
        # does not impact global seed
        random.Random(i).shuffle(shuffled)
        return [(i,x) for x in shuffled]
    
    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))
    
    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list(i)) for i in range(len(self.dataset_paths))])
    
    def __iter__(self):
        return self.get_streams()

        # batches = []
        # batch = []
        # for x in self.tokenizer_iter:
        #     batch.append(x)
        #     if len(batch) == self.batch_size:
        #         yield batch
        #         batch = []


class WebTextIter(IterableDataset):
    def __init__(self, batch_size, dataset_paths, seq_len, tokenizer=None, drop_last=True):
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.seq_len = seq_len
        self.dataset_paths = dataset_paths
        self.batch_size = batch_size
        self.batch_iter = BatchIterator(
            seq_len=seq_len,
            batch_size=batch_size,
            drop_last=drop_last,
            tokenizer=tokenizer,
            dataset_paths=dataset_paths,
        )

    def __iter__(self):
        try:
            batch = []
            for streams in self.batch_iter:
                for sample in streams:
                    if len(batch) == self.batch_size:
                        yield self.collate_fn(batch)
                        batch = []
                    batch.append(sample)

                # yield torch.LongTensor(src), torch.LongTensor(target), meta
        except StopIteration:
            return
            
    def collate_fn(self, batch):
        data_list, label_list, seq_len_list = [], [], []
        for _data, _label, _seq in batch:
            data_list.append(_data)
            label_list.append(_label)
            seq_len_list.append(_seq)
        return (
            torch.LongTensor(data_list),
            torch.LongTensor(label_list),
            seq_len_list,
        )

class LMOrderedIterator(IterableDataset):
    def __init__(self, data, length, bsz, bptt=None, device="cpu"):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.len = length
        self.bptt = bptt if bptt else bsz

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz
        self.data = data

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        # self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def __len__(self):
        return self.len

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = self.data.size(0) - 1 - i

        end_idx = i + seq_len
        beg_idx = i

        data = torch.LongTensor(self.data[beg_idx:end_idx])
        target = torch.LongTensor(self.data[i + 1 : i + 1 + seq_len])


        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(start)

    def __iter__(self):
        return self.get_fixlen_iter()


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        if self.dataset in ["ptb", "wikitext-2", "enwik8", "text8"]:
            self.vocab.count_file(os.path.join(path, "train.txt"))
            self.vocab.count_file(os.path.join(path, "valid.txt"))
            self.vocab.count_file(os.path.join(path, "test.txt"))
        elif self.dataset == "wikitext-103":
            self.vocab.count_file(os.path.join(path, "train.txt"))
        elif "states" in self.dataset:
            info_file = os.path.join(path, "info.txt")
            with open(info_file) as f:
                first_line = f.readline().strip()
                self.regex = first_line.replace("regex=", "")
            f.close()

            self.vocab.count_file(os.path.join(path, "0_shard_shuff.txt"))
            self.vocab.count_file(os.path.join(path, "1_shard_shuff.txt"))
            self.vocab.count_file(os.path.join(path, "2_shard_shuff.txt"))
        elif self.dataset == "openwebtext2":
            files = glob.glob(os.path.join(path + "/shards", "*"))
            train_paths = files[:80]
            valid_paths = files[80:90]
            test_paths = files[90:]

        self.vocab.build_vocab()

        if self.dataset in ["ptb", "wikitext-2", "wikitext-103"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train.txt"), ordered=True
            )
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=True
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=True
            )
        elif self.dataset in ["enwik8", "text8", "simple_wiki"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train.txt"), ordered=True, add_eos=False
            )
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=True, add_eos=False
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=True, add_eos=False
            )
        elif "states" in self.dataset:
            self.train = self.vocab.encode_file(
                os.path.join(path, "0_shard_shuff.txt"),
                ordered=True,
                add_bos_and_eos=True,
            )
            self.valid = self.vocab.encode_file(
                os.path.join(path, "1_shard_shuff.txt"),
                ordered=True,
                add_bos_and_eos=True,
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "2_shard_shuff.txt"),
                ordered=True,
                add_bos_and_eos=True,
            )
        elif "openwebtext2":
            self.train = train_paths
            self.valid = valid_paths
            self.test = test_paths

    def get_iterator(self, rank, world_size, split, batch_size, n_ctx, *args, **kwargs):
        if split == "train":
            if self.dataset in [
                "ptb",
                "wikitext-2",
                "wikitext-103",
                "enwik8",
                "text8",
            ]:
                data_iter = LMOrderedIterator(
                    self.train, len(self.train), batch_size, *args, **kwargs
                )
            elif "states" in self.dataset:
                data_iter = LMOrderedIterator(
                    self.train, len(self.train), batch_size, n_ctx, *args, **kwargs
                )
            elif self.dataset == "openwebtext2":
                n_partition = [
                    n for i, n in enumerate(self.train) if i % world_size == rank
                ]
                print("train partitions {}_{}".format(rank, len(n_partition)))

                dataset = WebTextIter(
                    batch_size=batch_size,
                    drop_last=True,
                    dataset_paths=n_partition,
                    seq_len=n_ctx,
                )
                return dataset

        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            if (
                self.dataset
                in ["ptb", "wikitext-2", "wikitext-103", "enwik8", "text8",]
                or "states" in self.dataset
            ):
                data_iter = LMOrderedIterator(
                    data, len(data), batch_size, n_ctx, *args, **kwargs
                )
            elif self.dataset == "openwebtext2":
                # data_iter = ConcatDataset(
                #     [
                #         WebTextDataset(data[i], n_ctx, self.vocab, *args, **kwargs)
                #         for i in range(len(data))
                #     ]
                # )
                print("eval partitions {}_{}".format(rank, len(data)))

                dataset = WebTextIter(
                    batch_size=batch_size,
                    drop_last=True,
                    dataset_paths=data,
                    seq_len=n_ctx,
                )

                return dataset

        dataloader = DataLoader(data_iter, batch_size=None, shuffle=False,)
        return dataloader


def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, "cache.pt")
    if os.path.exists(fn):
        print("Loading cached dataset...")
        corpus = torch.load(fn)
    else:
        print("Producing dataset {}...".format(dataset))
        kwargs = {}
        if dataset in ["wikitext-103", "wikitext-2"]:
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = False
        elif dataset == "ptb":
            kwargs["special"] = ["<unk>", "<eos>"]
            kwargs["lower_case"] = True
        elif dataset == "lm1b":
            kwargs["special"] = []
            kwargs["lower_case"] = False
            kwargs["vocab_file"] = os.path.join(datadir, "1b_word_vocab.txt")
        elif dataset in ["enwik8", "text8"]:
            pass
        elif dataset == "simple_wiki":
            kwargs["special"] = ["<eos>", "<bos>"]
            kwargs["delimiter"] = "\n"
        elif "states" in dataset:
            kwargs["special"] = ["<eos>", "<bos>"]
            kwargs["delimiter"] = ""
            kwargs["vocab_file"] = os.path.join(datadir, "vocab.txt")
        elif dataset == "openwebtext2":
            tokenizer = ByteLevelBPETokenizer().from_file(
                vocab_filename=datadir + "/gpt2-vocab.json",
                merges_filename=datadir + "/gpt2-merges.txt",
            )

            kwargs["tokenizer"] = tokenizer
            kwargs["vocab_file"] = os.path.join(datadir, "gpt2-vocab.json")

        corpus = Corpus(datadir, dataset, **kwargs)
        # torch.save(corpus, fn)

    return corpus


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")
    parser.add_argument(
        "--datadir",
        type=str,
        default="/datadrive/openwebtext2",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext2",
        choices=[
            "ptb",
            "wikitext-2",
            "wikitext-103",
            "lm1b",
            "enwik8",
            "text8",
            "openwebtext2",
        ],
        help="dataset name",
    )
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print("Vocab size : {}".format(len(corpus.vocab.idx2sym)))
