import datetime
import io
import json
import os
from collections import Counter
from collections import OrderedDict

import jsonlines
import torch
import zstandard


class Vocab(object):
    def __init__(
        self,
        special=[],
        min_freq=0,
        max_size=None,
        lower_case=True,
        delimiter=None,
        vocab_file=None,
        tokenizer=None,
    ):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        self.tokenizer = tokenizer

    def tokenize(
        self, line, add_bos_and_eos=False, add_eos=False, add_double_eos=False
    ):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == "":
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_double_eos:  # lm1b
            return ["<S>"] + symbols + ["<S>"]
        elif add_eos:
            return symbols + ["<EOS>"]
        elif add_bos_and_eos:
            return ["<BOS>"] + list(symbols) + ["<EOS>"]
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose:
            print("counting file {} ...".format(path))
        assert os.path.exists(path)

        sents = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print("    line {}".format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                if idx == 0:
                    print(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            print("counting {} sents ...".format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print("    line {}".format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        self.sym2idx["<UNK>"] = 0

        for sym in self.special:
            self.add_special(sym)

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx["<UNK>"]
        self.eos_idx = self.sym2idx["<eos>"]
        self.bos_idx = self.sym2idx["<bos>"]

    def _build_from_tokenizer(self):
        data = self.tokenizer.get_vocab()
        self.sym2idx = data
        self.idx2sym = {value:key for key, value in data.items()}

    def build_vocab(self):
        if self.vocab_file and self.tokenizer is None:
            print("building vocab from {}".format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print("final vocab size {}".format(len(self)))
            print("symbols: {}".format(list(self.sym2idx.keys())))
        elif self.tokenizer:
            print("building vocab from tokenizer")
            self._build_from_tokenizer()
            print("final vocab size {}".format(len(self)))
            print("symbols: {}".format(list(self.sym2idx.keys())))
        else:
            print(
                "building vocab with min_freq={}, max_size={}".format(
                    self.min_freq, self.max_size
                )
            )
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)
            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            print(
                "final vocab size {} from {} unique tokens".format(
                    len(self), len(self.counter)
                )
            )

    def encode_file(
        self,
        path,
        ordered=False,
        verbose=True,
        add_eos=True,
        add_bos_and_eos=False,
        add_double_eos=False,
    ):
        if verbose:
            print("encoding file {} ...".format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print("    line {}".format(idx))
                symbols = self.tokenize(
                    line,
                    add_eos=add_eos,
                    add_bos_and_eos=add_bos_and_eos,
                    add_double_eos=add_double_eos,
                )
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_zst(
        self,
        path,
        ordered=False,
        verbose=True,
    ):
        if verbose:
            print("encoding file {} ...".format(path))
        assert os.path.exists(path)
        encoded = []

        reader = Reader()
        idx = 0
        for document in reader.read_jsonl(path):
            if verbose and idx % 50000 == 0:
                print("    line {}".format(idx))
            symbols = self.tokenizer.encode(
                    document,
                ).tokens
            encoded.append(self.convert_to_tensor(symbols))
            idx += 1

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=True):
        if verbose:
            print("encoding {} sents ...".format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print("    line {}".format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, "{}_idx".format(sym.strip("<>")), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), "Index {} out of range".format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert "<eos>" not in sym
            assert hasattr(self, "unk_idx")
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return " ".join([self.get_sym(idx) for idx in indices])
        else:
            return " ".join(
                [self.get_sym(idx) for idx in indices if idx not in exclude]
            )

    def __len__(self):
        return len(self.idx2sym)


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
        self.fh = open(self.file_path, 'wb')
        self.cctx = zstandard.ZstdCompressor(level=compression_level)
        self.compressor = self.cctx.stream_writer(self.fh)

    def add_data(self, data, meta={}):
        self.compressor.write(json.dumps({'text': data, 'meta': meta}, default=json_serial).encode('UTF-8') + b'\n')

    def commit(self):
        self.compressor.flush(zstandard.FLUSH_FRAME)
        self.fh.flush()
        self.fh.close()


class Reader:
    def __init__(self):
        pass

    def read_jsonl(self, file, get_meta=False, autojoin_paragraphs=True, para_joiner='\n\n'):
        with open(file, 'rb') as fh:
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

                text = ob['text']

                if autojoin_paragraphs and isinstance(text, list):
                    text = para_joiner.join(text)

                if get_meta:
                    yield text, (ob['meta'] if 'meta' in ob else {})
                else:
                    yield text
