import datetime
import math
import os
import pickle
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pytz import timezone
from pytz import utc
from transformers import GPT2Tokenizer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

from datamodules import FileDataModule
from datamodules import OpenWebText2DataModule
from datamodules import WikiText2DataModule
from optim_utils import GradualWarmupScheduler
from optim_utils import CosineAnnealingWarmupRestarts

PICKLE_FILE = "/datadrive/shakespeare_output.pkl"


def get_pst_time():
    date_format = "%m_%d_%Y_%H_%M_%S_%Z"
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone("US/Pacific"))
    pstDateTime = date.strftime(date_format)
    return pstDateTime


def add_to_pickle(item, path=PICKLE_FILE):
    with open(path, "ab") as file:
        pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)


def get_trainer(args):
    if args.dataset == "wikitext2":
        print('getting wikitext2 datamodule')

        data_module = WikiText2DataModule(
            sequence_length=args.n_ctx, batch_size=args.mini_batch_size
        )
    elif args.dataset == "openwebtext2":
        print('getting openwebtext2 datamodule')

        data_module = OpenWebText2DataModule(
            sequence_length=args.n_ctx,
            batch_size=args.mini_batch_size,
            eval_batch_size=args.eval_batch_size,
            data_dir=args.data,
        )
    else:
        print('getting file datamodule')

        data_module = FileDataModule(
            sequence_length=args.n_ctx,
            batch_size=args.mini_batch_size,
            eval_batch_size=args.eval_batch_size,
            data_dir=args.data,
        )
    print('preparing dm')
    data_module.prepare_data()
    print('setting up dm')
    data_module.setup("fit")
    ntokens = len(data_module.vocab)
    args.n_tokens = ntokens

    # configuration = GPTConfig(
    #     vocab_size=args.n_tokens,
    #     context_length=args.n_ctx,
    #     n_embd=args.d_embd,
    #     n_layer=args.n_layer,
    #     n_head=args.n_head,
    #     d_ff=args.d_ff,
    # )
    # model = GPT(configuration)
    print('creating config')
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
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum(
        [p.nelement() for p in model.parameters() if p.requires_grad]
    )
    gpt_pl = GPTLightning(model=model, args=args, tokenizer = data_module.tokenizer)

    dt_string = get_pst_time()

    run_name = "{}_{}_{}_{}".format(args.dataset, args.model_size, args.note, dt_string)
    if args.local:
        print('is local')
        wandb_logger = WandbLogger(name=run_name, project="openwebtext2", entity=args.entity, log_model=True, save_dir='/datadrive/wandb')
    else:
        wandb_logger = WandbLogger(name=run_name, project="openwebtext2", entity=args.entity,)

    #if args.n_gpus > 1:

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
        trainer = pl.Trainer(
            val_check_interval=args.eval_interval * args.accumulate_grad_batches,
            weights_summary="full",
            gpus=[args.n_gpus],
            logger=wandb_logger,
            gradient_clip_val=args.clip,
            limit_val_batches=args.max_eval_steps,
            accumulate_grad_batches=args.accumulate_grad_batches,
            max_steps=args.max_step*10000,
            enable_pl_optimizer=True,
            log_every_n_steps=args.accumulate_grad_batches,
        )
    trainer.fit(gpt_pl, datamodule=data_module)


class GPTLightning(pl.LightningModule):
    def __init__(self, model, args, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.save_hyperparameters(args)
        self.train_seen = []
        self.val_seen = []
        self.tokenizer = tokenizer
        self.save_intervals = [x-1 for x in [10**1, 10**2, 10**3, 10**4, 10**5, 10**6]]

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward

        src, _ = batch

        outputs = self.model(input_ids=src, labels=src)
        loss = outputs[0]

        float_loss = loss.item()
        self.logger.experiment.log(
            {
                "loss": float_loss,
                "ppl": math.exp(float_loss),
                "bpc": (float_loss / math.log(2)),
                "tokens": (self.global_step) * self.args.batch_size * self.args.n_ctx,
                "lr": self.optimizers().param_groups[0]["lr"],
            },
            step=self.global_step,
        )




        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        src, meta = batch
        outputs = self.model(input_ids=src, labels=src)
        loss = outputs[0].item()

        # Add sync_dist=True to sync logging across all GPU workers
        # self.logger.log_metrics(
        #     {
        #         "validation_loss": loss,
        #         "validation_ppl": math.exp(loss),
        #         "validation_bpc": (loss / math.log(2)),
        #     },
        #     sync_dist=True,

        # )
        self.logger.experiment.log(
            {
                "validation_loss": loss,
                "validation_ppl": math.exp(loss),
                "validation_bpc": (loss / math.log(2)),
                "tokens": (self.global_step) * self.args.batch_size * self.args.n_ctx,
            },
            step=self.global_step,
        )



        return {"val_loss": outputs[0]}

    def validation_epoch_end(self, validation_step_outputs):
        epoch_metric = torch.mean(
            torch.stack([x["val_loss"] for x in validation_step_outputs])
        )
        tokens =  (self.global_step) * self.args.batch_size * self.args.n_ctx
        self.logger.experiment.log(
            {
                "validation_avg_loss": epoch_metric.item(),
                "validation_avg_ppl": math.exp(epoch_metric.item()),
                "validation_avg_bpc": (epoch_metric.item() / math.log(2)),
                "tokens": (self.global_step) * self.args.batch_size * self.args.n_ctx,
            },
            step=self.global_step,
        )
        
        if self.global_step in self.save_intervals:
            file_path = '{dir}/{step:02d}step-{token}token-{val_loss:.2f}loss.pt'.format(dir=self.logger.experiment.dir, step=self.global_step, token=tokens, val_loss=epoch_metric.item())
            torch.save({
                'step': self.global_step,
                'tokens': tokens,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizers[0].state_dict(),
                'validation_avg_loss': epoch_metric.item(),
            }, file_path)
            self.logger.experiment.save(file_path)
            outputs = self.model.generate(
                input_ids=None,
                do_sample=True,
                max_length=40,  # desired output sentence length
                pad_token_id=self.model.config.eos_token_id,
                bos_token_id=self.model.config.bos_token_id,
            )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.log("generated", generated, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, x_len = batch
        outputs = self.model(input_ids=x, labels=y)
        loss = outputs[0].item()
        # Add sync_dist=True to sync logging across all GPU workers
        self.log_dict(
            {
                "test_loss": loss,
                "test_ppl": math.exp(loss),
                "test_bpc": (loss / math.log(2)),
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {
                "test_loss": loss,
                "test_ppl": math.exp(loss),
                "test_bpc": (loss / math.log(2)),
            },
            step=self.global_step,
        )
        return outputs[0]

    def configure_optimizers(self):
        if self.args.optim.lower() == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError

        # #### scheduler
        # if self.args.scheduler == "cosine":
        #     cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer,
        #         self.args.max_steps,
        #         eta_min=.1*self.args.lr,
        #     )
        #     scheduler = GradualWarmupScheduler(
        #         optimizer, self.args.warmup_step, after_scheduler=cosine_scheduler
        #     )

        # elif self.args.scheduler == "inv_sqrt":
        #     # originally used for Transformer (in Attention is all you need)
        #     def lr_lambda(step):
        #         # return a multiplier instead of a learning rate
        #         if step == 0 and self.args.warmup_step == 0:
        #             return 1.0
        #         else:
        #             return (
        #                 1.0 / (step ** 0.5)
        #                 if step > self.args.warmup_step
        #                 else step / (self.args.warmup_step ** 1.5)
        #             )

        #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # else:
        #     raise NotImplementedError

        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=self.args.max_step, cycle_mult=1.0, max_lr=self.args.lr, min_lr=.1*self.args.lr, warmup_steps=self.args.warmup_step, gamma=1.0)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "val_loss",
                }
            ],
        )
