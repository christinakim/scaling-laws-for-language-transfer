import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

from datamodules import ChineseWebtextDataModule
from datamodules import FileDataModule
from datamodules import OpenWebText2DataModule
from datamodules import OscarDataModule
from gpt import GPTLightning
from utils import get_pst_time


def get_trainer(args):
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
    elif "oscar" in args.dataset:
        print("getting Oscar datamoduble")
        data_module = OscarDataModule(
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
    if args.finetune > 0:
        print("finetuning")
        checkpoint_file = "{}/{}.pt".format(args.checkpoints_dir, args.model_size)
        checkpoint = torch.load(
            checkpoint_file, map_location="cuda:{}".format(args.n_gpus)
        )
        state_dict = checkpoint["model_state_dict"]

        model.load_state_dict(state_dict)

    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum(
        [p.nelement() for p in model.parameters() if p.requires_grad]
    )
    gpt_pl = GPTLightning(model=model, args=args, tokenizer=data_module.tokenizer)

    dt_string = get_pst_time()

    run_name = "{}_{}_{}_{}".format(args.dataset, args.model_size, args.note, dt_string)

    wandb_logger = WandbLogger(
        name=run_name,
        project="openwebtext2",
        entity=args.entity,
        save_dir=args.save_dir,
    )

    eval_interval = (
        args.eval_interval * args.accumulate_grad_batches
        if args.eval_interval > 0
        else 1.0
    )
    limit_train_batches = (
        args.limit_train_batches * args.accumulate_grad_batches
        if args.limit_train_batches > 0
        else 1.0
    )
    print("eval interval is {}".format(eval_interval))
    if args.n_gpus > 1:
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
                EarlyStopping(monitor="validation_avg_loss", patience=3, verbose=True)
            ],
            limit_train_batches=limit_train_batches,
            limit_val_batches=args.max_eval_steps,
            # check_val_every_n_epoch=1,
        )
    trainer.fit(gpt_pl, datamodule=data_module)
    file_path = "{dir}/last.pt".format(dir=wandb_logger.experiment.dir,)
    trainer.save_checkpoint(file_path)

    #use latest model and run on test set
    trainer.test(ckpt_path=None, datamodule=data_module)
