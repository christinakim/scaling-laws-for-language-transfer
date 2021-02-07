# coding: utf-8
import argparse
import math
import os
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from data_utils import get_lm_corpus
from gpt import GPT
from gpt import GPTConfig
from gpt import common_models_by_name
from trainer import Trainer
from utils.exp_utils import create_exp_dir


def main(args):

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



    if args.d_embd < 0:
        args.d_embd = args.d_model

    assert args.batch_size % args.batch_chunk == 0
    args.work_dir = "{}-{}".format(args.work_dir, args.dataset)
    args.work_dir = os.path.join(args.work_dir, time.strftime("%Y%m%d-%H%M%S"))
    logger = create_exp_dir(
        args.work_dir, scripts_to_save=["train.py",], debug=args.debug,
    )

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )
        else:
            torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    world_size = args.n_gpus
    args.lr = args.lr 
    args.batch_size = args.batch_size // world_size

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
    args.n_nonemb_param = sum(
        [p.nelement() for p in model.parameters() if p.requires_grad]
    )

    if args.restart:
        if os.path.exists(os.path.join(args.restart_dir, "optimizer.pt")):
            with open(os.path.join(args.restart_dir, "optimizer.pt"), "rb") as f:
                opt_state_dict = torch.load(f)
                optimizer.load_state_dict(opt_state_dict)
        else:
            print("Optimizer was not saved. Start from scratch.")

    logger("=" * 100)
    for k, v in args.__dict__.items():
        logger("    - {} : {}".format(k, v))
    logger("=" * 100)
    logger("#params = {}".format(args.n_all_param))
    logger("#non emb params = {}".format(args.n_nonemb_param))
    ###############################################################################
    # Training code
    ###############################################################################


    # At any point you can hit Ctrl + C to break out of training early.
    trainer = Trainer(
        model=model, logger=logger, corpus=corpus, args=args, device=device, 
    )
    try:
        mp.spawn(trainer.train, args=(world_size,), nprocs=world_size, join=True)

    except KeyboardInterrupt:
        dist.destroy_process_group()
        logger("-" * 100)
        logger("Exiting from training early")

    # Load the best saved model.

    with open(os.path.join(args.work_dir, "model.pt"), "rb") as f:
        loaded_model = torch.load(f)
    model = loaded_model.to(device)

    # Run on test data.
    test_iter = corpus.get_iterator(0, 1, "test", args.batch_size, args.n_ctx,)
    test_loss, test_tokens = trainer.test(model, test_iter)
    logger("=" * 100)
    logger(
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

    logger("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch GPT Model")
    parser.add_argument(
        "--data",
        type=str,
        default="../data/wikitext-103",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext-103", help="dataset name"
    )
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
    parser.add_argument(
        "--n_layer", type=int, default=12, help="number of total layers"
    )
    parser.add_argument("--n_head", type=int, default=10, help="number of heads")
    parser.add_argument("--d_embd", type=int, default=-1, help="embedding dimension")
    parser.add_argument("--d_model", type=int, default=500, help="model dimension")
    parser.add_argument("--d_ff", type=int, default=1000, help="inner dimension in FF")
    parser.add_argument("--n_ctx", type=int, default=128, help="context length")
    parser.add_argument("--n_positions", type=int, default=500, help="max seq length")
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="global dropout rate"
    )
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
        "--lr", type=float, default=-1, help="initial learning rate",
    )
    parser.add_argument(
        "--scheduler",
        default="cosine",
        type=str,
        choices=["cosine", "inv_sqrt", "constant"],
        help="lr scheduler to use.",
    )
    parser.add_argument(
        "--warmup_step", type=int, default=3000, help="upper epoch limit"
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=0.0,
        help="minimum learning rate during annealing",
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
        "--batch_chunk",
        type=int,
        default=1,
        help="split batch into chunks to save memory",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--adaptive", action="store_true", help="use adaptive softmax")

    parser.add_argument("--log-interval", type=int, default=10, help="report interval")
    parser.add_argument(
        "--eval_interval", type=int, default=1000, help="evaluation interval"
    )
    parser.add_argument(
        "--work_dir", default="experiments", type=str, help="experiment directory."
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart training from the saved checkpoint",
    )
    parser.add_argument("--restart_dir", type=str, default="", help="restart dir")
    parser.add_argument(
        "--debug", action="store_true", help="run in debug mode (do not create exp dir)"
    )
    parser.add_argument(
        "--same_length",
        action="store_true",
        help="use the same attn length for all tokens",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=0.0,
        help="min learning rate for cosine scheduler",
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
    parser.add_argument("--n_nodes", default=1, type=int, metavar="N")
    parser.add_argument("--n_gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")
    args = parser.parse_args()
    main(args)
