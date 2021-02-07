import glob
import os
from pathlib import Path

import itertools
import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from utils.vocabulary import Vocab


class WebTextDataset(Dataset):
    def __init__(
        self, dataset_path, seq_len, vocab,
    ):
        self.vocab = vocab
        data = list(itertools.chain.from_iterable(self.vocab.encode_zst(dataset_path)))[:1000]
        stop = (len(data)//seq_len) * seq_len
        self.data = data[:stop]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, i):
        end_idx = i + self.seq_len
        beg_idx = i
        
        data_pre_tensor= self.data[beg_idx:end_idx]

        data =  torch.LongTensor(data_pre_tensor)
        target =  torch.LongTensor(self.data[i + 1 : i + 1 + self.seq_len])
        return data, target, self.seq_len

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
        #self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def __len__(self):
        return self.len

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = self.data.size(0) - 1 - i

        end_idx = i + seq_len
        beg_idx = i

        data = torch.LongTensor(self.data[beg_idx:end_idx])
        target =  torch.LongTensor(self.data[i + 1 : i + 1 + seq_len])

        print(data)

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
            all_paths = [str(x) for x in Path(path).glob("**/*.zst")]
            train_paths = [path for idx, path in enumerate(all_paths) if idx % 10 in (0,2, 4, 6, 8,)]
            valid_paths = [path for idx, path in enumerate(all_paths) if idx % 10 in (1, 9 )]
            test_paths = [path for idx, path in enumerate(all_paths) if idx % 10 in (3, 7 )]

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

    def get_iterator(self, rank, world_size, split,  batch_size, n_ctx, *args, **kwargs):
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
                data_iter = LMOrderedIterator(self.train, len(self.train), batch_size, n_ctx, *args, **kwargs)
            elif self.dataset == "openwebtext2":
                n_partition = [n for i, n in enumerate(self.train[:1]) if i % world_size == rank]
                print("{}_{}".format(rank, len(n_partition)))

                data_iter = ConcatDataset([WebTextDataset(data, n_ctx, self.vocab, *args, **kwargs) for data in n_partition])
                dataloader = DataLoader(data_iter, batch_size=batch_size, shuffle=False, drop_last=True)
                return dataloader

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
                data_iter = ConcatDataset([WebTextDataset(data[i], n_ctx, self.vocab, *args, **kwargs) for i in range(len(data[:1]))])
                dataloader = DataLoader(data_iter, batch_size=batch_size, shuffle=False, drop_last=True)

                return dataloader
        
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
            tokenizer = ByteLevelBPETokenizer().from_file(vocab_filename=datadir + '/gpt2-vocab.json',
                                merges_filename=datadir + "/gpt2-merges.txt")

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
        choices=["ptb", "wikitext-2", "wikitext-103", "lm1b", "enwik8", "text8", "openwebtext2"],
        help="dataset name",
    )
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print("Vocab size : {}".format(len(corpus.vocab.idx2sym)))
