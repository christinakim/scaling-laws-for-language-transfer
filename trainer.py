import datetime
import itertools
import math
import os
import time

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

from data_utils import get_lm_corpus
from datamodules import OpenWebText2DataModule
from datamodules import WikiText2DataModule
from utils.sample import sample_words


def get_trainer(args):
    if args.dataset == "wikitext2":
        data_module = WikiText2DataModule(
            sequence_length=args.n_ctx, batch_size=args.batch_size
        )
    elif args.dataset == "openwebtext2":
        data_module = OpenWebText2DataModule(
            sequence_length=args.n_ctx, batch_size=args.batch_size,
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
    configuration = GPT2Config(
        vocab_size=args.n_tokens,
        n_ctx=args.n_ctx,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.d_ff,
    )
    model = GPT2LMHeadModel(configuration)
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum(
        [p.nelement() for p in model.parameters() if p.requires_grad]
    )
    gpt_pl = GPTLightning(model=model, args=args)

    run_name = "{}_{}".format(args.dataset, args.model_size)
    wandb_logger = WandbLogger(name=run_name, project=args.dataset, entity=args.entity,)
    if args.n_gpus >= 1:
        trainer = pl.Trainer(
            val_check_interval=args.eval_interval,
            weights_summary="full",
            gpus=args.n_gpus,
            logger=wandb_logger,
            accelerator="ddp",
            gradient_clip_val=args.clip,
        )
    else:
        trainer = pl.Trainer(
            val_check_interval=args.eval_interval,
            weights_summary="full",
            gpus=args.n_gpus,
            logger=wandb_logger,
            gradient_clip_val=args.clip,
        )
    trainer.fit(gpt_pl, datamodule=data_module)


class GPTLightning(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.args = args
        self.model = model
        self.save_hyperparameters(args)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y, x_len = batch
        output = self.model(input_ids=x, labels=y)
        loss = output[0].item()
        self.log_dict(
            {
                "loss": loss,
                "ppl": math.exp(loss),
                "bpc": (loss / math.log(2)),
                # "tokens": self.global_step * x_len,
            },
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.logger.experiment.log(
            {
                "loss": loss,
                "ppl": math.exp(loss),
                "bpc": (loss / math.log(2)),
            },
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_len = batch
        output = self.model(input_ids=x, labels=y)
        loss = output[0].item()

        # Add sync_dist=True to sync logging across all GPU workers
        self.log_dict(
            {
                "validation_loss": loss,
                "validation_ppl": math.exp(loss),
                "validation_bpc": (loss / math.log(2)),
            },
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {
                "validation_loss": loss,
                "validation_ppl": math.exp(loss),
                "validation_bpc": (loss / math.log(2)),
            },
        )

        return loss

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
        )
        return loss

    def configure_optimizers(self):
        if self.args.optim.lower() == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError

        #### scheduler
        if self.args.scheduler == "cosine":
            # here we do not set eta_min to lr_min to be backward compatible
            # because in previous versions eta_min is default to 0
            # rather than the default value of lr_min 1e-6
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.args.max_step, eta_min=self.args.eta_min
            )  # should use eta_min arg

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

        return [optimizer], [scheduler]


class Trainer:
    def __init__(
        self, model, logger, corpus, args, device,
    ):
        self.model = model

        self.args = args
        self.early_stop = False
        self.train_step = 0
        self.best_val_loss = None
        self.early_stop = False
        self.logger = logger
        self.corpus = corpus
        self.log_start_time = time.time()
        self.eval_start_time = time.time()

    def configure_optimizers(self, model, args):
        if self.args.optim.lower() == "adam":
            if self.args.sample_softmax > 0:
                dense_params, sparse_params = [], []
                for param in model.parameters():
                    if param.size() == model.word_emb.weight.size():
                        sparse_params.append(param)
                    else:
                        dense_params.append(param)
                optimizer = optim.Adam(dense_params, lr=self.args.lr)
            else:
                optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError

        #### scheduler
        if self.args.scheduler == "cosine":
            # here we do not set eta_min to lr_min to be backward compatible
            # because in previous versions eta_min is default to 0
            # rather than the default value of lr_min 1e-6
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.args.max_step, eta_min=self.args.eta_min
            )  # should use eta_min arg

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
        elif self.args.scheduler == "constant":
            pass

        return optimizer, scheduler

    def init_process(self, rank, size, backend="gloo"):
        """ Initialize the distributed environment. """
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(
            backend,
            rank=rank,
            world_size=size,
            timeout=datetime.timedelta(0, seconds=10),
        )

    def cleanup(self):
        dist.destroy_process_group()

    def get_model(self):
        return self.model

    def evaluate(self, model, val_iter, rank):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        # Evaluation
        total_len, total_loss = 0, 0.0
        with torch.no_grad():
            for i, (data, target, seq_len) in enumerate(val_iter):
                if self.args.max_eval_steps > 0 and i >= self.args.max_eval_steps:
                    break
                data = data.to(rank)
                target = target.to(rank)
                seq_len = torch.sum(seq_len.to(rank))
                logits, loss = model(data, target)
                loss = loss.mean()
                total_loss += seq_len * loss.float()
                total_len += seq_len
        # Switch back to the training mode
        model.train()
        return total_loss / total_len, total_len

    def test(self, model, test_iter):
        # Turn on evaluation mode which disables dropout.
        model = model.to(0)
        model.eval()

        # Evaluation
        total_len, total_loss = 0, 0.0
        with torch.no_grad():
            for i, (data, target, seq_len) in enumerate(test_iter):
                if self.args.max_eval_steps > 0 and i >= self.args.max_eval_steps:
                    break
                data = data.to(0)
                target = target.to(0)
                logits, loss = model(data, target)
                loss = loss.mean()
                total_loss += seq_len * loss.float()
                total_len += seq_len

        return total_loss / total_len, total_len

    def train(self, rank, world_size):
        self.init_process(rank, world_size)
        self.logger(f"{rank + 1}/{world_size} process initialized.\n")
        if rank == 0 and self.args.wandb:
            if "states" in self.args.dataset:
                wandb.init(
                    project="regular-language-scaling-laws", entity=self.args.entity
                )
            else:
                wandb.init(
                    project="natural-language-scaling-laws", entity=self.args.entity
                )
            wandb.run.name = "{}_{}".format(self.args.dataset, self.args.model_size)
            wandb.config.update(self.args)

        self.logger(
            f"Rank {rank}/{world_size} training process passed data download barrier.\n"
        )

        model = self.model.to(rank)

        model = DistributedDataParallel(model, device_ids=[rank])

        self.logger("getting iterators")
        self.train_iter = self.corpus.get_iterator(
            rank, world_size, "train", self.args.batch_size, self.args.n_ctx,
        )
        self.val_iter = self.corpus.get_iterator(
            rank, world_size, "valid", self.args.batch_size, self.args.n_ctx,
        )

        self.logger("getting optimizers")
        optimizer, scheduler = self.configure_optimizers(model, self.args)

        # Turn on training mode which enables dropout.
        model.train()
        for epoch in itertools.count(start=1):
            self.logger("starting epoch {} on device {}".format(epoch, rank))
            self.train_epoch(
                rank, epoch, model, optimizer, scheduler, wandb,
            )
            if (
                (self.args.max_epoch and self.args.max_epoch == epoch)
                or self.train_step == self.args.max_step
                or self.early_stop
            ):
                self.logger("-" * 100)
                self.logger("End of training")
                break
        self.cleanup()

    def train_epoch(
        self, rank, epoch, model, optimizer, scheduler, wandb,
    ):
        self.logger("inside epoch")
        train_loss = 0
        n_val_no_improve = 0
        for batch_idx, (data, target, seq_len) in enumerate(self.train_iter):
            data = data.to(rank)
            target = target.to(rank)
            logits, loss = model(data, target)

            model.zero_grad()

            loss.backward()
            train_loss += loss.float().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)

            optimizer.step()

            # step-wise learning rate annealing
            self.train_step += 1
            if self.args.scheduler in [
                "cosine",
                "constant",
            ]:
                # linear warmup stage
                if self.train_step < self.args.warmup_step:
                    curr_lr = self.args.lr * self.train_step / self.args.warmup_step
                    optimizer.param_groups[0]["lr"] = curr_lr
                else:
                    if self.args.scheduler == "cosine":
                        scheduler.step()
            elif self.args.scheduler == "inv_sqrt":
                scheduler.step()

            if self.train_step % self.args.log_interval == 0:
                if rank == 0:
                    cur_loss = train_loss / self.args.log_interval
                    elapsed = time.time() - self.log_start_time
                    log_str = (
                        "| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} "
                        "| ms/batch {:5.2f} | loss {:5.2f}".format(
                            epoch,
                            self.train_step,
                            batch_idx + 1,
                            optimizer.param_groups[0]["lr"],
                            elapsed * 1000 / self.args.log_interval,
                            cur_loss,
                        )
                    )

                    log_str += " | bpc {:9.5f}".format(cur_loss / math.log(2))
                    log_str += " | ppl {:9.3f}".format(math.exp(cur_loss))
                    self.logger(log_str)
                    if self.args.wandb:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "ppl": math.exp(cur_loss),
                                "loss": cur_loss,
                                "bpc": cur_loss / math.log(2),
                                "tokens": (batch_idx + 1) * self.args.n_ctx,
                                "lr": optimizer.param_groups[0]["lr"],
                            },
                        )

                train_loss = 0
                self.log_start_time = time.time()
            if self.train_step % self.args.eval_interval == 0:
                if rank == 0:

                    val_loss, total_tokens = self.evaluate(model, self.val_iter, rank)
                    print(val_loss, total_tokens)

                    self.logger("-" * 100)
                    log_str = (
                        "| Eval {:3d} at step {:>8d} | time: {:5.2f}s "
                        "| valid loss {:5.2f}".format(
                            self.train_step // self.args.eval_interval,
                            self.train_step,
                            (time.time() - self.eval_start_time),
                            val_loss.item(),
                        )
                    )
                    log_str += " | bpc {:9.5f}".format(val_loss / math.log(2))
                    log_str += " | valid ppl {:9.3f}".format(math.exp(val_loss))

                    if self.args.wandb:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "val_loss": val_loss,
                                "val_ppl": math.exp(val_loss),
                                "val_bpc": (val_loss / math.log(2)),
                                "val_tokens": total_tokens
                                * (self.train_step // self.args.eval_interval),
                                "eval_step": self.train_step // self.args.eval_interval,
                            },
                        )
                    self.logger(log_str)
                    self.logger("-" * 100)
                    # Save the model if the validation loss is the best we've seen so far.
                    if not self.best_val_loss or val_loss < self.best_val_loss:
                        if not self.args.debug and rank == 0:
                            # only the main node gets to do i/o ops
                            raw_model = (
                                model.module if hasattr(model, "module") else model
                            )
                            self.logger("saving new model")

                            with open(
                                os.path.join(self.args.work_dir, "model.pt"), "wb"
                            ) as f:

                                torch.save(raw_model, f)
                            with open(
                                os.path.join(self.args.work_dir, "optimizer.pt"), "wb"
                            ) as f:
                                torch.save(optimizer.state_dict(), f)
                        self.best_val_loss = val_loss
                        n_val_no_improve = 0
                    else:
                        n_val_no_improve += 1
                        self.logger(
                            "validation loss hasn't improved in {} evals".format(
                                n_val_no_improve
                            )
                        )

                    self.eval_start_time = time.time()

                    if n_val_no_improve == self.args.n_val_stop:
                        self.logger("early stopping due to val loss not decreasing")
                        self.early_stop = False

            if (
                "states" in self.args.dataset
                and self.args.sample
                and self.train_step % self.args.sample_interval == 0
            ) and rank == 0:
                # sample here
                words = sample_words(
                    model,
                    100,
                    self.corpus.vocab.sym2idx,
                    self.corpus.vocab.idx2sym,
                    device=self.device,
                    temperature=1.0,
                    sample=True,
                    top_k=10,
                )
                good_samples = list(filter(self.corpus.regex_compiled.match, words))
                self.logger("-" * 100)
                self.logger(good_samples)
                self.logger("sample accuracy {:3d}".format(len(good_samples)))
                self.logger("-" * 100)

                if self.args.wandb:
                    wandb.log({"sample_accuracy": len(good_samples)})

            if self.train_step == self.args.max_step or self.early_stop:
                if self.early_stop and self.args.wandb:
                    wandb.run.summary["early_stop"] = self.train_step
                break
        if self.early_stop:
            self.logger("-" * 100)
            self.logger("early stopping of training")
            return
