# coding: utf-8
import argparse
import re
import time
import math
import os, sys
import itertools

import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from transformers import OpenAIGPTConfig
from transformers import OpenAIGPTLMHeadModel
from transformers import OpenAIGPTModel

from gpt import GPTConfig
from gpt import GPT
from data_utils import get_lm_corpus
from gpt import ModelSettings
from gpt import common_models_by_name
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from utils.sample import sample
from utils.sample import sample_words

parser = argparse.ArgumentParser(description="PyTorch GPT Model")
parser.add_argument(
    "--data",
    type=str,
    default="../data/wikitext-103",
    help="location of the data corpus",
)
parser.add_argument("--dataset", type=str, default="wt103", help="dataset name")
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
# parser.add_argument('--init', default='normal', type=str,
#                     help='parameter initializer to use.')
# parser.add_argument('--emb_init', default='normal', type=str,
#                     help='parameter initializer to use.')
# parser.add_argument('--init_range', type=float, default=0.1,
#                     help='parameters initialized by U(-init_range, init_range)')
# parser.add_argument('--emb_init_range', type=float, default=0.01,
#                     help='parameters initialized by U(-init_range, init_range)')
# parser.add_argument('--init_std', type=float, default=0.02,
#                     help='parameters initialized by N(0, init_std)')
# parser.add_argument('--proj_init_std', type=float, default=0.01,
#                     help='parameters initialized by N(0, init_std)')
parser.add_argument(
    "--optim",
    default="adam",
    type=str,
    choices=["adam", "sgd", "adagrad"],
    help="optimizer to use.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.00025,
    help="initial learning rate (0.00025|5 for adam|sgd)",
)
parser.add_argument("--mom", type=float, default=0.0, help="momentum for sgd")
parser.add_argument(
    "--scheduler",
    default="cosine",
    type=str,
    choices=["cosine", "inv_sqrt", "dev_perf", "constant"],
    help="lr scheduler to use.",
)
parser.add_argument("--warmup_step", type=int, default=3000, help="upper epoch limit")
parser.add_argument(
    "--decay_rate",
    type=float,
    default=0.5,
    help="decay factor when ReduceLROnPlateau is used",
)
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


parser.add_argument(
    "--not_tied",
    action="store_true",
    help="do not tie the word embedding and softmax weights",
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--adaptive", action="store_true", help="use adaptive softmax")
parser.add_argument(
    "--div_val",
    type=int,
    default=1,
    help="divident value for adapative input and softmax",
)
parser.add_argument(
    "--pre_lnorm",
    action="store_true",
    help="apply LayerNorm to the input instead of the output",
)
parser.add_argument("--varlen", action="store_true", help="use variable length")
parser.add_argument("--multi_gpu", action="store_true", help="use multiple GPU")
parser.add_argument("--log-interval", type=int, default=10, help="report interval")
parser.add_argument(
    "--eval-interval", type=int, default=1000, help="evaluation interval"
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
    "--clamp_len",
    type=int,
    default=-1,
    help="use the same pos embeddings after clamp_len",
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
parser.add_argument("--patience", type=int, default=0, help="patience")
parser.add_argument("--finetune_v2", action="store_true", help="finetune v2")
parser.add_argument("--finetune_v3", action="store_true", help="finetune v3")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Run in pseudo-fp16 mode (fp16 storage fp32 math).",
)
parser.add_argument(
    "--static-loss-scale",
    type=float,
    default=1,
    help="Static loss scale, positive power of 2 values can "
    "improve fp16 convergence.",
)
parser.add_argument(
    "--dynamic-loss-scale",
    action="store_true",
    help="Use dynamic loss scaling.  If supplied, this argument"
    " supersedes --static-loss-scale.",
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

args = parser.parse_args()

if not args.wandb:
    #if we aren't logging then we don't need to save
    args.debug = True

if args.wandb:
    wandb.init(project="regular language")

if args.model_size:
    print('model config of size {}'.format(args.model_size))
    config = common_models_by_name.get(args.model_size)
    args.n_layer = config.n_layer
    args.d_model = config.d_model
    args.lr = config.learning_rate
    args.n_head = config.n_head
    args.d_ff = config.d_ff
    args.d_attn = config.d_attn

if args.d_embd < 0:
    args.d_embd = args.d_model

assert args.batch_size % args.batch_chunk == 0

args.work_dir = "{}-{}".format(args.work_dir, args.dataset)
args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
logging = create_exp_dir(args.work_dir, scripts_to_save=["train.py",], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    if not args.cuda:
        print("WARNING: --fp16 requires --cuda, ignoring --fp16 option")
        args.fp16 = False
    else:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print("WARNING: apex not installed, ignoring --fp16 option")
            args.fp16 = False

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_tokens = ntokens
args.regex = corpus.regex
regex_compiled = re.compile(str(args.regex))

eval_batch_size = 2
tr_iter = corpus.get_iterator("train", args.batch_size, args.n_ctx, device=device,)
va_iter = corpus.get_iterator("valid", eval_batch_size, args.n_ctx, device=device,)
te_iter = corpus.get_iterator("test", eval_batch_size, args.n_ctx, device=device,)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ["wt103", "lm1b"]
    if args.dataset == "wt103":
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == "lm1b":
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)

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

if args.fp16:
    model = model.half()

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(
            args.gpu0_bsz // args.batch_chunk, model, dim=1
        ).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)

#### optimizer
if args.optim.lower() == "sgd":
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
elif args.optim.lower() == "adam":
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
elif args.optim.lower() == "adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

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
elif args.scheduler == "dev_perf":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min
    )
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_sparse,
            factor=args.decay_rate,
            patience=args.patience,
            min_lr=args.lr_min,
        )
elif args.scheduler == "constant":
    pass

if args.cuda and args.fp16:
    # If args.dynamic_loss_scale is False, static_loss_scale will be used.
    # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
    optimizer = FP16_Optimizer(
        optimizer,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
        dynamic_loss_args={"init_scale": 2 ** 16},
    )

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
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()

    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    for batch, (data, target, seq_len) in enumerate(train_iter):
        logits, loss = para_model(data, target)
        model.zero_grad()

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ["cosine", "constant", "dev_perf"]:
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
                        "tokens": batch + 1 * args.n_ctx,
                    }
                )
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss, total_tokens = evaluate(va_iter)
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
                        "val_loss": loss,
                        "val_ppl": math.exp(val_loss),
                        "val_bpc": (val_loss / math.log(2)),
                        "val_tokens": total_tokens * (train_step//args.eval_interval),
                    }
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


            eval_start_time = time.time()
        if args.sample and train_step % args.sample_interval == 0:
            # sample here
            words = sample_words(model, 100, corpus.vocab.sym2idx, corpus.vocab.idx2sym,  device=device, temperature=1.0, sample=True, top_k=10)
            good_samples = list(filter(regex_compiled.match, words))
            logging("-" * 100)
            logging(good_samples)
            logging("sample accuracy {:3d}".format(len(good_samples)))
            logging("-" * 100)

            if args.wandb:
                wandb.log({
                    "sample_accuracy": len(good_samples)
                })
        if train_step == args.max_step:
            break


# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        if (args.max_epoch and args.max_epoch == epoch) or train_step == args.max_step:
            logging("-" * 100)
            logging("End of training")
            break
except KeyboardInterrupt:
    logging("-" * 100)
    logging("Exiting from training early")

# Load the best saved model.
with open(os.path.join(args.work_dir, "model.pt"), "rb") as f:
    model = torch.load(f)
para_model = model.to(device)

# Run on test data.
test_loss, test_tokens = evaluate(te_iter)
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

logging("=" * 100)
