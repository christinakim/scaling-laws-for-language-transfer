import glob
import os

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.distributed import DistributedSampler

from utils.vocabulary import Vocab


class LMOrderedIterator(IterableDataset):
    def __init__(self, data, len, bsz, bptt, device="cpu"):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.len = len

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt
    def __len__(self):
        return self.len
    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = i

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()



class LMShuffledIterator(IterableDataset):
    def __init__(self, data, bsz, bptt, device="cpu", shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = (
            np.random.permutation(len(self.data))
            if self.shuffle
            else np.array(range(len(self.data)))
        )

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[
                            n_retain + n_filled : n_retain + n_filled + n_new, i
                        ] = streams[i][:n_new]
                        target[n_filled : n_filled + n_new, i] = streams[i][
                            1 : n_new + 1
                        ]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = data.size(0)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(
        self, paths, vocab, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.paths = paths
        self.vocab = vocab


    def get_sent_stream_from_path(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream_from_path(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


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
        elif self.dataset == "lm1b":
            train_path_pattern = os.path.join(
                path,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called
        elif "states" in self.dataset:
            info_file = os.path.join(path, "info.txt")
            with open(info_file) as f:
                first_line = f.readline().strip()
                self.regex = first_line.replace("regex=", "")
            f.close()

            self.vocab.count_file(os.path.join(path, "0_shard_shuff.txt"))
            self.vocab.count_file(os.path.join(path, "1_shard_shuff.txt"))
            self.vocab.count_file(os.path.join(path, "2_shard_shuff.txt"))

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
        elif self.dataset == "lm1b":
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=False, add_double_eos=True
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=False, add_double_eos=True
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

    def get_iterator(self, rank, world_size, split, *args, **kwargs):
        if split == "train":
            if self.dataset in [
                "ptb",
                "wikitext-2",
                "wikitext-103",
                "enwik8",
                "text8",
            ]:
                data_iter = LMOrderedIterator(self.train, len(self.train) ,*args, **kwargs)
            elif self.dataset == "lm1b":
                kwargs["shuffle"] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
            elif "states" in self.dataset:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)

        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            if (
                self.dataset
                in ["ptb", "wikitext-2", "wikitext-103", "enwik8", "text8",]
                or "states" in self.dataset
            ):
                data_iter = LMOrderedIterator(data, len(data), *args, **kwargs)
            elif self.dataset == "lm1b":
                data_iter = LMShuffledIterator(data, *args, **kwargs)

        sampler = DistributedSampler(data_iter, rank=rank, num_replicas=world_size)
        dataloader = DataLoader(data_iter, batch_size=args.batch_size, shuffle=False, sampler=sampler)
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
            kwargs["special"] = ["<unk>","<eos>"]
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
        else:
            kwargs["special"] = ["<eos>", "<bos>"]
            kwargs["delimiter"] = ""
            kwargs["vocab_file"] = os.path.join(datadir, "vocab.txt")

        corpus = Corpus(datadir, dataset, **kwargs)
        # torch.save(corpus, fn)

    return corpus


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unit test")
    parser.add_argument(
        "--datadir",
        type=str,
        default="../data/text8",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="text8",
        choices=["ptb", "wikitext-2", "wikitext-103", "lm1b", "enwik8", "text8"],
        help="dataset name",
    )
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print("Vocab size : {}".format(len(corpus.vocab.idx2sym)))
