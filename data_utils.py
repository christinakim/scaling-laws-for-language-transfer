import datetime
import io
import json
import os
import random
from itertools import chain
from itertools import cycle

import jsonlines
import torch
import zstandard
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset
from transformers import GPT2Tokenizer


# from openwebtext2 https://github.com/EleutherAI/openwebtext2/blob/master/utils/archiver.py
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime,)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class Archive:
    def __init__(self, file_path, compression_level=3):
        self.file_path = file_path
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.fh = open(self.file_path, "wb")
        self.cctx = zstandard.ZstdCompressor(level=compression_level)
        self.compressor = self.cctx.stream_writer(self.fh)

    def add_data(self, data, meta={}):
        self.compressor.write(
            json.dumps({"text": data, "meta": meta}, default=json_serial).encode(
                "UTF-8"
            )
            + b"\n"
        )

    def commit(self):
        self.compressor.flush(zstandard.FLUSH_FRAME)
        self.fh.flush()
        self.fh.close()


class Reader:
    def __init__(self):
        pass

    def read_jsonl(
        self, file, get_meta=False, autojoin_paragraphs=True, para_joiner="\n\n"
    ):
        with open(file, "rb") as fh:
            self.fh = fh
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            for ob in rdr:
                # naive jsonl where each object is just the string itself, with no meta. For legacy compatibility.
                if isinstance(ob, str):
                    assert not get_meta
                    yield ob
                    continue

                text = ob["text"]

                if autojoin_paragraphs and isinstance(text, list):
                    text = para_joiner.join(text)

                if get_meta:
                    yield file, text, (ob["meta"] if "meta" in ob else {})
                else:
                    yield file, text


def collate_fn(batch):
    data_list, label_list, seq_len_list = [], [], []
    # for _data, _label, _seq in batch:
    for _data, _seq in batch:
        data_list.append(_data)
        seq_len_list.append(_seq)
    return (
        torch.LongTensor(data_list),
        # torch.LongTensor(label_list),
        seq_len_list,
    )


class WebTextDocumentIterator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __iter__(self):
        reader = Reader()
        doc_chunk_size = 20000
        documents = []
        for i, x in enumerate(reader.read_jsonl(self.dataset_path)):
            documents.append(x)
            if len(documents) == doc_chunk_size:
                yield documents
                documents = []
        yield documents


class FileIterator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __iter__(self):
        line_chunk_size = 20000
        lines = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
            yield text


class TokenizerIterator:
    def __init__(self, seq_len, tokenizer, seed, dataset_path):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.document_iter = WebTextDocumentIterator(dataset_path)

        self.seed = seed

    def __iter__(self):
        block = []
        if self.document_iter:
            for documents in self.document_iter:
                random.Random(self.seed).shuffle(documents)
                docs = []
                for doc_i, x in enumerate(documents):
                    tokenized = self.tokenizer(text=x[1],).input_ids
                    tokenized.append(self.tokenizer.eos_token_id)

                    tokenized.insert(0, self.tokenizer.eos_token_id)
                    tokenized_length = len(tokenized)
                    for token in tokenized:
                        if len(block) == self.seq_len:
                            yield block, (x[0], doc_i)
                            block = []
                        block.append(token)
        else:
            for file in self.file_iter:
                tokenized = self.tokenizer(text=file,).input_ids

                tokenized_length = len(tokenized)
                idxs = [i for i in range(tokenized_length - self.seq_len)]
                while len(idxs) > 1:
                    starting_idx = idxs.pop(random.randrange(len(idxs)))
                    block = tokenized[starting_idx : starting_idx + self.seq_len]
                    yield block, (starting_idx)

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
        self.tokenizer_iter = TokenizerIterator(
            self.seq_len, self.tokenizer, seed, dataset
        )
        for x in self.tokenizer_iter:
            yield x
            # batch.append(x)
            # if len(batch) == self.batch_size:
            #     yield batch
            #     batch = []

    def shuffled_data_list(self, i):
        # split = len(self.dataset_paths) // self.batch_size
        # dataset_paths = self.dataset_paths[(i*split):((i+1)*split)]
        shuffled = self.dataset_paths
        # does not impact global seed
        random.Random(i).shuffle(shuffled)
        return [(i, x) for x in shuffled]

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(
            *[
                self.get_stream(self.shuffled_data_list(i))
                for i in range(len(self.dataset_paths))
            ]
        )

    def __iter__(self):
        return self.get_streams()


class WebTextIter(IterableDataset):
    def __init__(
        self, batch_size, dataset_paths, seq_len, tokenizer=None, drop_last=True
    ):
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
                        yield collate_fn(batch)
                        batch = []
                    batch.append(sample)

        except StopIteration:
            return


class ChineseWebtextDataset(IterableDataset):
    def __init__(
        self,
        file: str,
        seq_len: int,
        batch_size: int,
        token_limit: int,
        tokenizer=None,
    ):
        self.file = file
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_limit = token_limit
        self.token_count = 0
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            self.tokenizer = tokenizer

    def get_block(self):
        block = []
        with jsonlines.open(self.file) as reader:
            for obj in reader:
                content = obj["content"]
                tokenized = self.tokenizer(text=content,).input_ids
                tokenized.append(self.tokenizer.eos_token_id)
                tokenized.insert(0, self.tokenizer.eos_token_id)
                for token in tokenized:
                    if len(block) == self.seq_len:
                        yield block, len(block)
                        self.token_count += len(block)
                        block = []
                    block.append(token)

    def __iter__(self):
        batch = []
        for x in self.get_block():
            batch.append(x)
            if len(batch) == self.batch_size:
                yield collate_fn(batch)
                if 0 < self.token_limit <= self.token_count:
                    return
                batch = []