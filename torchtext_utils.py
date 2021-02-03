import torchtext
from torchtext.datasets import EnWik9
from torchtext.experimental.datasets import PennTreebank

PennTreebank
tokenizer = ByteLevelBPETokenizer().from_file(
    vocab_filename=datadir + "/gpt2-vocab.json",
    merges_filename=datadir + "/gpt2-merges.txt",
)
EnWik9
pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
TEXT = data.Field(use_vocab=False, tokenize=tokenizer.encode, pad_token=pad_index)
