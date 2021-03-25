import math

import pytorch_lightning as pl
import torch
from torch import optim as optim

from optim_utils import CosineAnnealingWarmupRestarts


class ModelSettings:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(
        self, size: str, n_layer: int, d_model: int, learning_rate: float,
    ):
        self.size = size
        self.n_layer = n_layer
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.n_head = max(2, self.d_model // 64)
        self.d_ff = 4 * d_model
        self.d_attn = 1 * d_model


common_models_by_name = {
    "x10small": ModelSettings(
        size="x10small", n_layer=1, d_model=8, learning_rate=0.00211,
    ),
    "x9small": ModelSettings(
        size="x9small", n_layer=1, d_model=16, learning_rate=0.00211,
    ),
    "x8small": ModelSettings(
        size="x8small", n_layer=1, d_model=32, learning_rate=0.00211,
    ),
    "x7small": ModelSettings(
        size="x7small", n_layer=2, d_model=32, learning_rate=0.00211,
    ),
    "x6small": ModelSettings(
        size="x6small", n_layer=2, d_model=64, learning_rate=0.00211,
    ),
    "x5small": ModelSettings(
        size="x5small", n_layer=2, d_model=128, learning_rate=0.00202,
    ),
    "x4small": ModelSettings(
        size="x4small", n_layer=4, d_model=256, learning_rate=0.00173,
    ),
    "x3small": ModelSettings(
        size="x3small", n_layer=4, d_model=512, learning_rate=0.00163,
    ),
    "x2small": ModelSettings(
        size="x2small", n_layer=8, d_model=512, learning_rate=0.00144,
    ),
    "x1small": ModelSettings(
        size="x1small", n_layer=6, d_model=768, learning_rate=0.00146,
    ),
    "small": ModelSettings(
        size="small", n_layer=12, d_model=768, learning_rate=0.0006,
    ),
    "medium": ModelSettings(
        size="medium", n_layer=24, d_model=1024, learning_rate=0.0003,
    ),
    "large": ModelSettings(
        size="large", n_layer=24, d_model=1536, learning_rate=0.00025,
    ),
    "xl": ModelSettings(size="xl", n_layer=24, d_model=2048, learning_rate=0.00000625,),
}


class GPTLightning(pl.LightningModule):
    def __init__(self, model, args, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.save_hyperparameters(args)
        self.train_seen = []
        self.val_seen = []
        self.tokenizer = tokenizer
        intervals = [i for i in range(10 ** 4, 100000, 15000)]
        intervals.extend(
            [10, 50, 10 ** 2, 10 ** 3, 10 ** 4,]
        )
        self.save_intervals = [x - 1 for x in intervals]

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
                "tokens": (self.global_step + 1)
                * self.args.batch_size
                * self.args.n_ctx,
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
                "tokens": (self.global_step + 1)
                * self.args.batch_size
                * self.args.n_ctx,
            },
            step=self.global_step,
        )

        return {"val_loss": outputs[0]}

    def validation_epoch_end(self, validation_step_outputs):
        epoch_metric = torch.mean(
            torch.stack([x["val_loss"] for x in validation_step_outputs])
        )
        tokens = (self.global_step) * self.args.batch_size * self.args.n_ctx
        self.logger.experiment.log(
            {
                "validation_avg_loss": epoch_metric.item(),
                "validation_avg_ppl": math.exp(epoch_metric.item()),
                "validation_avg_bpc": (epoch_metric.item() / math.log(2)),
                "tokens": (self.global_step + 1)
                * self.args.batch_size
                * self.args.n_ctx,
            },
            step=self.global_step,
        )

        # if self.global_step in self.save_intervals:
        #     file_path = "{dir}/{step:02d}step-{token}token-{val_loss:.2f}loss.pt".format(
        #         dir=self.logger.experiment.dir,
        #         step=self.global_step,
        #         token=tokens,
        #         val_loss=epoch_metric.item(),
        #     )
        #     torch.save(
        #         {
        #             "step": self.global_step,
        #             "tokens": tokens,
        #             "model_state_dict": self.model.state_dict(),
        #             "optimizer_state_dict": self.trainer.optimizers[0].state_dict(),
        #             "validation_avg_loss": epoch_metric.item(),
        #         },
        #         file_path,
        #     )
        #     self.logger.experiment.save(file_path)
        # outputs = self.model.generate(
        #     input_ids=None,
        #     do_sample=True,
        #     max_length=40,  # desired output sentence length
        #     pad_token_id=self.model.config.eos_token_id,
        #     bos_token_id=self.model.config.bos_token_id,
        # )

        # generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # self.log("generated", generated, prog_bar=True)
        self.log("validation_avg_loss", epoch_metric.item())

    def test_step(self, batch, batch_idx):
        src, meta = batch
        outputs = self.model(input_ids=src, labels=src)
        loss = outputs[0].item()

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

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.args.max_step,
            cycle_mult=1.0,
            max_lr=self.args.lr,
            min_lr=0.1 * self.args.lr,
            warmup_steps=self.args.warmup_step,
            gamma=1.0,
        )
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
