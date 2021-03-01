import datetime
import math
import pickle
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from pytz import timezone
from pytz import utc
from transformers import GPT2Tokenizer
from transformers import OpenAIGPTConfig
from transformers import OpenAIGPTLMHeadModel

from datamodules import OpenWebText2DataModule
from datamodules import WikiText2DataModule
from optim_utils import GradualWarmupScheduler

PICKLE_FILE = "/datadrive/batches_9.pkl"


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
        data_module = WikiText2DataModule(
            sequence_length=args.n_ctx, batch_size=args.mini_batch_size
        )
    elif args.dataset == "openwebtext2":
        data_module = OpenWebText2DataModule(
            sequence_length=args.n_ctx,
            batch_size=args.mini_batch_size,
            eval_batch_size=args.eval_batch_size,
            data_dir=args.data,
        )
    else:
        raise NotImplementedError

    data_module.prepare_data()
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
    configuration = OpenAIGPTConfig(
        vocab_size=args.n_tokens,
        n_ctx=args.n_ctx,
        n_positions=args.n_ctx,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.d_ff,
        n_embd=args.d_embd,
    )

    model = OpenAIGPTLMHeadModel(configuration)
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum(
        [p.nelement() for p in model.parameters() if p.requires_grad]
    )
    gpt_pl = GPTLightning(model=model, args=args)

    dt_string = get_pst_time()

    run_name = "{}_{}_{}".format(args.dataset, args.model_size, dt_string)
    wandb_logger = WandbLogger(name=run_name, project=args.dataset, entity=args.entity)

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
            val_check_interval=args.eval_interval * args.accumulate_grad_batches,
            weights_summary="full",
            gpus=[1],
            logger=wandb_logger,
            gradient_clip_val=args.clip,
            limit_val_batches=args.max_eval_steps,
            accumulate_grad_batches=args.accumulate_grad_batches,
            max_steps=args.max_step,
            enable_pl_optimizer=True,
            log_every_n_steps=args.accumulate_grad_batches,
        )
    trainer.fit(gpt_pl, datamodule=data_module)


class GPTLightning(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.args = args
        self.model = model
        self.save_hyperparameters(args)
        self.train_seen = []
        self.val_seen = []
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward

        src, target, _ = batch

        add_to_pickle(batch)

        outputs = self.model(input_ids=src, labels=target)
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

        src, target, meta = batch
        outputs = self.model(input_ids=src, labels=target)
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

        self.logger.experiment.log(
            {
                "validation_avg_loss": epoch_metric.item(),
                "validation_avg_ppl": math.exp(epoch_metric.item()),
                "validation_avg_bpc": (epoch_metric.item() / math.log(2)),
                "tokens": (self.global_step) * self.args.batch_size * self.args.n_ctx,
            },
            step=self.global_step,
        )
        sentence_prefix = "I am"

        input_ids = self.tokenizer.encode(
            sentence_prefix, add_special_tokens=False, return_tensors="pt",
        ).to(self.device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_length=20,  # desired output sentence length
            pad_token_id=self.model.config.eos_token_id,
        )[0].tolist()

        generated_text = self.tokenizer.decode(
            output_ids, clean_up_tokenization_spaces=True
        )
        self.log("generated", generated_text, prog_bar=True)

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

        #### scheduler
        if self.args.scheduler == "cosine":
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.trainer.max_steps // self.args.batch_size,
                eta_min=self.args.lr * 0.2,
            )
            scheduler = GradualWarmupScheduler(
                optimizer, self.args.warmup_step, after_scheduler=cosine_scheduler
            )

        elif self.args.scheduler == "inv_sqrt":
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and self.args.warmup_step == 0:
                    return 1.0
                else:
                    return (
                        1.0 / (step ** 0.5)
                        if step > self.args.warmup_step
                        else step / (self.args.warmup_step ** 1.5)
                    )

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        else:
            raise NotImplementedError
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
