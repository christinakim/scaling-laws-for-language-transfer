# coding: utf-8
import argparse
import itertools
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from data_utils import get_lm_corpus
from gpt import GPT
from gpt import GPTConfig
from gpt import common_models_by_name
from utils.data_parallel import BalancedDataParallel
from utils.exp_utils import create_exp_dir
from utils.sample import sample_words

parser = argparse.ArgumentParser(description="PyTorch GPT Model")
parser.add_argument(
    "--data",
    type=str,
    default="../data/wikitext-103",
    help="location of the data corpus",
)
parser.add_argument("--dataset", type=str, default="wikitext-103", help="dataset name")
parser.add_argument(
    "--model_size",
    type=str,
    choices=[
        "x10small",
        "x9small",
        "x8small",
        "x7small",
        "x6small",
        "x5small",
        "x4small",
        "x3small",
        "x2small",
        "small",
        "medium",
        "large",
    ],
    help="model size",
)
parser.add_argument("--n_layer", type=int, default=12, help="number of total layers")
parser.add_argument("--n_head", type=int, default=10, help="number of heads")
parser.add_argument("--d_embd", type=int, default=-1, help="embedding dimension")
parser.add_argument("--d_model", type=int, default=500, help="model dimension")
parser.add_argument("--d_ff", type=int, default=1000, help="inner dimension in FF")
parser.add_argument("--n_ctx", type=int, default=128, help="context length")
parser.add_argument("--n_positions", type=int, default=500, help="max seq length")
parser.add_argument("--dropout", type=float, default=0.0, help="global dropout rate")
parser.add_argument(
    "--dropatt", type=float, default=0.0, help="attention probability dropout rate"
)
parser.add_argument(
    "--optim",
    default="adam",
    type=str,
    choices=["adam", "sgd", "adagrad"],
    help="optimizer to use.",
)
parser.add_argument(
    "--lr", type=float, default=0.00025, help="initial learning rate",
)
parser.add_argument(
    "--scheduler",
    default="cosine",
    type=str,
    choices=["cosine", "inv_sqrt", "constant"],
    help="lr scheduler to use.",
)
parser.add_argument("--warmup_step", type=int, default=3000, help="upper epoch limit")
parser.add_argument(
    "--lr_min", type=float, default=0.0, help="minimum learning rate during annealing"
)
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument(
    "--clip_nonemb",
    action="store_true",
    help="only clip the gradient of non-embedding params",
)
parser.add_argument("--max_step", type=int, default=100000, help="upper step limit")
parser.add_argument("--max_epoch", type=int, help="upper epoch limit")
parser.add_argument("--batch_size", type=int, default=60, help="batch size")
parser.add_argument(
    "--batch_chunk", type=int, default=1, help="split batch into chunks to save memory"
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--adaptive", action="store_true", help="use adaptive softmax")

parser.add_argument("--varlen", action="store_true", help="use variable length")
parser.add_argument("--multi_gpu", action="store_true", help="use multiple GPU")
parser.add_argument("--log-interval", type=int, default=10, help="report interval")
parser.add_argument(
    "--eval_interval", type=int, default=1000, help="evaluation interval"
)
parser.add_argument(
    "--work_dir", default="experiments", type=str, help="experiment directory."
)
parser.add_argument(
    "--restart", action="store_true", help="restart training from the saved checkpoint"
)
parser.add_argument("--restart_dir", type=str, default="", help="restart dir")
parser.add_argument(
    "--debug", action="store_true", help="run in debug mode (do not create exp dir)"
)
parser.add_argument(
    "--same_length", action="store_true", help="use the same attn length for all tokens"
)
parser.add_argument(
    "--eta_min", type=float, default=0.0, help="min learning rate for cosine scheduler"
)
parser.add_argument("--gpu0_bsz", type=int, default=-1, help="batch size on gpu 0")
parser.add_argument("--max_eval_steps", type=int, default=-1, help="max eval steps")
parser.add_argument(
    "--sample_softmax",
    type=int,
    default=-1,
    help="number of samples in sampled softmax",
)
parser.add_argument(
    "--wandb", action="store_false", help="Log to wandb if absent",
)
parser.add_argument(
    "--sample", action="store_true", help="if included sample",
)
parser.add_argument(
    "--sample-interval", type=int, default=1000, help="sample interval",
)
parser.add_argument("--entity", type=str, default="openai-scholars")
parser.add_argument("--n_val_stop", type=int, default=3)

args = parser.parse_args()

if not args.wandb:
    # if we aren't logging then we don't need to save
    args.debug = True

if args.model_size:
    print("model config of size {}".format(args.model_size))
    config = common_models_by_name.get(args.model_size)
    args.n_layer = config.n_layer
    args.d_model = config.d_model
    if args.lr < 0:
        args.lr = config.learning_rate
    args.n_head = config.n_head
    args.d_ff = config.d_ff
    args.d_attn = config.d_attn

if args.wandb:
    if "states" in args.dataset:
        wandb.init(project="regular-language-scaling-laws", entity=args.entity)
    else:
        wandb.init(project="natural-language-scaling-laws", entity=args.entity)
    wandb.run.name = "{}_{}".format(args.dataset, args.model_size)

if args.d_embd < 0:
    args.d_embd = args.d_model

assert args.batch_size % args.batch_chunk == 0

args.work_dir = "{}-{}".format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
logging = create_exp_dir(
    args.work_dir, scripts_to_save=["train.py",], debug=args.debug,
)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_tokens = ntokens
if "states" in args.dataset:
    args.regex = corpus.regex
    regex_compiled = re.compile(str(args.regex))

eval_batch_size = 5
train_iter = corpus.get_iterator("train", args.batch_size, args.n_ctx, device=device,)
val_iter = corpus.get_iterator("valid", eval_batch_size, args.n_ctx, device=device,)
test_iter = corpus.get_iterator("test", eval_batch_size, args.n_ctx, device=device,)

###############################################################################
# Build the model
###############################################################################
configuration = GPTConfig(
    vocab_size=args.n_tokens,
    context_length=args.n_ctx,
    n_embd=args.d_embd,
    n_layer=args.n_layer,
    n_head=args.n_head,
    d_ff=args.d_ff,
)
model = GPT(configuration)

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

model = model.to(device)
if args.multi_gpu:
    model = nn.DataParallel(model).to(device)

if args.optim.lower() == "adam":
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError

#### scheduler
if args.scheduler == "cosine":
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_step, eta_min=args.eta_min
    )  # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_sparse, args.max_step, eta_min=args.eta_min
        )  # should use eta_min arg
elif args.scheduler == "inv_sqrt":
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.0
        else:
            return (
                1.0 / (step ** 0.5)
                if step > args.warmup_step
                else step / (args.warmup_step ** 1.5)
            )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == "constant":
    pass

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, "optimizer.pt")):
        with open(os.path.join(args.restart_dir, "optimizer.pt"), "rb") as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print("Optimizer was not saved. Start from scratch.")

logging("=" * 100)
for k, v in args.__dict__.items():
    logging("    - {} : {}".format(k, v))
logging("=" * 100)
logging("#params = {}".format(args.n_all_param))
logging("#non emb params = {}".format(args.n_nonemb_param))
###############################################################################
# Training code
###############################################################################
if args.wandb:
    wandb.config.update(args)


def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.

    # Evaluation
    total_len, total_loss = 0, 0.0
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            logits, loss = model(data, target)
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.train()

    return total_loss / total_len, total_len

def train():
    # Turn on training mode which enables dropout.
    global early_stop, train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()

    for batch, (data, target, seq_len) in enumerate(train_iter):
        logits, loss = model(data, target)
        model.zero_grad()

        loss.backward()
        train_loss += loss.float().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in [
            "cosine",
            "constant",
        ]:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]["lr"] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]["lr"] = curr_lr * 2
            else:
                if args.scheduler == "cosine":
                    scheduler.step()
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == "inv_sqrt":
            scheduler.step()

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = (
                "| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} "
                "| ms/batch {:5.2f} | loss {:5.2f}".format(
                    epoch,
                    train_step,
                    batch + 1,
                    optimizer.param_groups[0]["lr"],
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                )
            )

            log_str += " | bpc {:9.5f}".format(cur_loss / math.log(2))
            log_str += " | ppl {:9.3f}".format(math.exp(cur_loss))
            logging(log_str)
            if args.wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "ppl": math.exp(cur_loss),
                        "loss": cur_loss,
                        "bpc": cur_loss / math.log(2),
                        "tokens": (batch + 1) * args.n_ctx,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss, total_tokens = evaluate(val_iter)
            logging("-" * 100)
            log_str = (
                "| Eval {:3d} at step {:>8d} | time: {:5.2f}s "
                "| valid loss {:5.2f}".format(
                    train_step // args.eval_interval,
                    train_step,
                    (time.time() - eval_start_time),
                    val_loss,
                )
            )
            log_str += " | bpc {:9.5f}".format(val_loss / math.log(2))
            log_str += " | valid ppl {:9.3f}".format(math.exp(val_loss))

            if args.wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_ppl": math.exp(val_loss),
                        "val_bpc": (val_loss / math.log(2)),
                        "val_tokens": total_tokens * (train_step // args.eval_interval),
                        "eval_step": train_step // args.eval_interval,
                    },
                )
            logging(log_str)
            logging("-" * 100)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    logging("saving new model")
                    with open(os.path.join(args.work_dir, "model.pt"), "wb") as f:

                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, "optimizer.pt"), "wb") as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss
                n_val_no_improve = 0
            else:
                n_val_no_improve += 1
                logging("validation loss hasn't improved in {} evals".format(n_val_no_improve))

            eval_start_time = time.time()

            if n_val_no_improve == args.n_val_stop:
                logging("early stopping due to val loss not decreasing")
                early_stop = True

        if (
            "states" in args.dataset
            and args.sample
            and train_step % args.sample_interval == 0
        ):
            # sample here
            words = sample_words(
                model,
                100,
                corpus.vocab.sym2idx,
                corpus.vocab.idx2sym,
                device=device,
                temperature=1.0,
                sample=True,
                top_k=10,
            )
            good_samples = list(filter(regex_compiled.match, words))
            logging("-" * 100)
            logging(good_samples)
            logging("sample accuracy {:3d}".format(len(good_samples)))
            logging("-" * 100)

            if args.wandb:
                wandb.log({"sample_accuracy": len(good_samples)})
        if train_step == args.max_step or early_stop:
            break


# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None
early_stop = False

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        if (args.max_epoch and args.max_epoch == epoch) or train_step == args.max_step or early_stop:
            logging("-" * 100)
            logging("End of training")
            break
except KeyboardInterrupt:
    logging("-" * 100)
    logging("Exiting from training early")


# Load the best saved model.
with open(os.path.join(args.work_dir, "model.pt"), "rb") as f:
    loaded_model = torch.load(f)
model = loaded_model.to(device)

# Run on test data.
test_loss, test_tokens = evaluate(test_iter)
logging("=" * 100)
logging(
    "| End of training | test loss {:5.2f} | test bpc {:9.5f}| test ppl {:9.3f}".format(
        test_loss, test_loss / math.log(2), math.exp(test_loss)
    )
)
if args.wandb:
    wandb.log(
        {
            "test_loss": test_loss,
            "test_ppl": math.exp(test_loss),
            "test_bpc": (test_loss / math.log(2)),
            "test_tokens": test_tokens,
        }
    )

if early_stop and args.wandb:
    wandb.run.summary['early_stop'] = train_step
else:
    wandb.run.summary['early_stop'] = -1

logging("=" * 100)
