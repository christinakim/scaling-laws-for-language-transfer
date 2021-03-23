# coding: utf-8
# coding: utf-8
import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

from utils import get_pst_time
from datamodules import ChineseWebtextDataModule
from datamodules import FileDataModule
from datamodules import OpenWebText2DataModule
from gpt import GPTLightning
from gpt import common_models_by_name


def get_test_trainer(args):
    if args.dataset == "openwebtext2":
        print("getting openwebtext2 datamodule")

        data_module = OpenWebText2DataModule(
            sequence_length=args.n_ctx,
            batch_size=args.mini_batch_size,
            eval_batch_size=args.eval_batch_size,
            data_dir=args.data,
        )
    elif args.dataset == "webtext2019zh":
        print("getting webtext2019zh datamodule")

        data_module = ChineseWebtextDataModule(
            sequence_length=args.n_ctx,
            batch_size=args.mini_batch_size,
            eval_batch_size=args.eval_batch_size,
            data_dir=args.data,
            token_limit=args.token_limit,
            diff_tokenization=True if args.diff_tokenization > 0 else False,
        )
    else:
        print("getting file datamodule")

        data_module = FileDataModule(
            sequence_length=args.n_ctx,
            batch_size=args.mini_batch_size,
            eval_batch_size=args.eval_batch_size,
            data_dir=args.data,
        )
    print("preparing dm")
    data_module.prepare_data()
    print("setting up dm")
    data_module.setup("fit")
    ntokens = len(data_module.vocab)
    args.n_tokens = ntokens

    print("creating config")
    configuration = GPT2Config(
        vocab_size=args.n_tokens,
        n_ctx=args.n_ctx,
        n_positions=args.n_ctx,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.d_ff,
        n_embd=args.d_embd,
        bos_token_id=data_module.tokenizer.bos_token_id,
        eos_token_id=data_module.tokenizer.eos_token_id,
        attn_pdrop=args.dropatt,
        embd_pdrop=args.dropout,
        resid_pdrop=args.dropout,
    )

    model = GPT2LMHeadModel(configuration)
    if args.model_size in ["x6small", "x5small"]:
        print("finetuning")
        checkpoint_file = "{}/{}.pt".format(args.checkpoints_dir, args.model_size)
        checkpoint = torch.load(checkpoint_file, map_location="cuda:0")
        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)
    else:
        checkpoint_file = "{}/{}.ckpt".format(args.checkpoints_dir, args.model_size)
        checkpoint = torch.load(checkpoint_file, map_location="cuda:0")
        new_state = {}
        state_dict = checkpoint["state_dict"]

        for key, value in state_dict.items():
           new_state[key[6:]] = value
        model.load_state_dict(new_state)

    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum(
        [p.nelement() for p in model.parameters() if p.requires_grad]
    )
    gpt_pl = GPTLightning(model=model, args=args, tokenizer=data_module.tokenizer)

    dt_string = get_pst_time()

    run_name = "{}_{}_{}_{}".format(args.dataset, args.model_size, args.note, dt_string)
    if args.local:
        print("is local")
        wandb_logger = WandbLogger(
            name=run_name,
            project="openwebtext2",
            entity=args.entity,
            save_dir="/datadrive/wandb",
        )
    else:
        wandb_logger = WandbLogger(
            name=run_name, project="openwebtext2", entity=args.entity,
        )

    # if args.n_gpus > 1:

    if False:
        trainer = pl.Trainer(
            val_check_interval=args.eval_interval,
            weights_summary="full",
            gpus=args.n_gpus,
            logger=wandb_logger,
            accelerator="ddp",
            gradient_clip_val=args.clip,
            limit_val_batches=args.max_eval_steps * args.accumulate_grad_batches,
            max_steps=args.max_step,
            accumulate_grad_batches=args.accumulate_grad_batches,
            enable_pl_optimizer=True,
        )
    else:
        print("no ddp")
        eval_interval = (
            args.eval_interval * args.accumulate_grad_batches
            if args.eval_interval > 0
            else 1.0
        )
        print("eval interval is {}".format(eval_interval))
        trainer = pl.Trainer(
            val_check_interval=eval_interval,
            weights_summary="full",
            gpus=[args.n_gpus],
            logger=wandb_logger,
            gradient_clip_val=args.clip,
            accumulate_grad_batches=args.accumulate_grad_batches,
            max_steps=args.max_step * 10000,
            max_epochs=1000,
            enable_pl_optimizer=True,
            log_every_n_steps=args.accumulate_grad_batches,
            callbacks=[
                EarlyStopping(monitor="validation_avg_loss", patience=5, verbose=True)
            ],
            limit_train_batches=args.limit_train_batches * args.accumulate_grad_batches,
            limit_val_batches=args.max_eval_steps,
            # check_val_every_n_epoch=1,
        )
    trainer.test(model=model, datamodule=data_module)


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

    if args.model_size in ["x2small", "small"]:
        args.mini_batch_size = 2
    else:
        args.mini_batch_size = 8

    args.accumulate_grad_batches = args.batch_size // args.mini_batch_size
    args.dataset_size = args.limit_train_batches * args.batch_size * args.n_ctx

    if args.d_embd < 0:
        args.d_embd = args.d_model

    assert args.batch_size % args.mini_batch_size == 0

    print(args)
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###############################################################################
    # eval code
    ###############################################################################
    try:
        get_test_trainer(args)

    except KeyboardInterrupt:
        print("YOOO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch GPT Model")
    parser.add_argument(
        "--data",
        type=str,
        default="/datadrive/webtext2019zh",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--dataset", type=str, default="webtext2019zh", help="dataset name"
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
        default="x6small",
    )
    parser.add_argument(
        "--n_layer", type=int, default=12, help="number of total layers"
    )
    parser.add_argument("--n_head", type=int, default=10, help="number of heads")
    parser.add_argument("--d_embd", type=int, default=-1, help="embedding dimension")
    parser.add_argument("--d_model", type=int, default=500, help="model dimension")
    parser.add_argument("--d_ff", type=int, default=1000, help="inner dimension in FF")
    parser.add_argument("--n_ctx", type=int, default=1024, help="context length")
    parser.add_argument("--n_positions", type=int, default=1024, help="max seq length")
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
    parser.add_argument("--warmup_step", type=int, default=2, help="upper epoch limit")
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
    parser.add_argument("--max_step", type=int, default=20, help="upper step limit")
    parser.add_argument("--max_epoch", type=int, help="upper epoch limit")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="batch size")
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
        "--eval_interval", type=int, default=2, help="evaluation interval"
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
    parser.add_argument("--max_eval_steps", type=int, default=10, help="max eval steps")
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
        "--local", action="store_true", help="remote if absent",
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
    parser.add_argument("--n_gpus", default=3, type=int, help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")
    parser.add_argument("--note", default="", type=str, help="run description")
    parser.add_argument("--token_limit", default=-1, type=int, help="for finetuning")
    parser.add_argument(
        "--finetune", default=-1, type=int, help="if included finetuning"
    )
    parser.add_argument(
        "--diff_tokenization", default=-1, type=int, help="if included diff token"
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="/datadrive/checkpoints/",
        type=str,
        help="finetune dff token",
    )
    parser.add_argument("--limit_train_batches", type=int, help="num of batches")
    args = parser.parse_args()
    main(args)
