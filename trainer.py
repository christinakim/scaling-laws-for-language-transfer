import itertools
import math
import os
import time

import torch
import wandb
from torch.nn.parallel.distributed import DistributedDataParallel

from utils.sample import sample_words
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# coding: utf-8

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import wandb


class Trainer:
    def __init__(
        self,
        model,
        logger,
        corpus,
        args,
        device,
        
    ):
        self.non_ddp_model = model

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

        return optimizer, scheduler

    def init_process(self, rank, size, backend='gloo'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)

    def cleanup(self):
        dist.destroy_process_group()

    def get_model(self):
        return self.non_ddp_model

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
                logits, loss = model(data, target)
                loss = loss.mean()
                total_loss += seq_len.item() * loss.float().item()
                total_len += seq_len.item()

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
                total_loss += seq_len.item() * loss.float().item()
                total_len += seq_len.item()

        return total_loss / total_len, total_len

    def train(self, rank, world_size):
        self.init_process(rank, world_size)
        self.logger(
            f"{rank + 1}/{world_size} process initialized.\n"
        )
        if rank == 0:
            self.corpus.get_iterator(rank, world_size, "train", self.args.batch_size, self.args.n_ctx,)
            self.corpus.get_iterator(rank, world_size, "valid", self.args.batch_size, self.args.n_ctx,)
            self.corpus.get_iterator(rank, world_size, "test", self.args.batch_size, self.args.n_ctx, )
            self.get_model()
        dist.barrier()

        self.logger(f"Rank {rank}/{world_size} training process passed data download barrier.\n")


        model = self.get_model().to(rank)
        
        model = DistributedDataParallel(model, device_ids=[rank])
        train_iter = self.corpus.get_iterator(rank, world_size, "train", self.args.batch_size, self.args.n_ctx, )
        val_iter = self.corpus.get_iterator(rank, world_size, "valid", self.args.batch_size, self.args.n_ctx, )

        optimizer, scheduler = self.configure_optimizers(model, self.args)

        # Turn on training mode which enables dropout.
        model.train()
        for epoch in itertools.count(start=1):
            self.train_epoch(rank, epoch, model, optimizer, scheduler, train_iter, val_iter)
            if (
                (self.args.max_epoch and self.args.max_epoch == epoch)
                or self.train_step == self.args.max_step
                or self.early_stop
        ):
                self.logger("-" * 100)
                self.logger("End of training")
                break

    def train_epoch(self, rank, epoch, model, optimizer, scheduler, train_iter, val_iter):
        train_loss = 0
        n_val_no_improve = 0
        for batch, (data, target, seq_len) in enumerate(train_iter):
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
                cur_loss = train_loss / self.args.log_interval
                elapsed = time.time() - self.log_start_time
                if rank == 0:
                    log_str = (
                        "| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} "
                        "| ms/batch {:5.2f} | loss {:5.2f}".format(
                            epoch,
                            self.train_step,
                            batch + 1,
                            optimizer.param_groups[0]["lr"],
                            elapsed * 1000 / self.args.log_interval,
                            cur_loss,
                        )
                    )

                    log_str += " | bpc {:9.5f}".format(cur_loss / math.log(2))
                    log_str += " | ppl {:9.3f}".format(math.exp(cur_loss))
                    self.logger(log_str)
                    if self.args.wandb:
                        self.wandb.log(
                            {
                                "epoch": epoch,
                                "ppl": math.exp(cur_loss),
                                "loss": cur_loss,
                                "bpc": cur_loss / math.log(2),
                                "tokens": (batch + 1) * self.args.n_ctx,
                                "lr": optimizer.param_groups[0]["lr"],
                            },
                        )
                train_loss = 0
                self.log_start_time = time.time()

            if self.train_step % self.args.eval_interval == 0:
                val_loss, total_tokens = self.evaluate(model, val_iter, rank)
                if rank == 0:
                    self.logger("-" * 100)
                    log_str = (
                        "| Eval {:3d} at step {:>8d} | time: {:5.2f}s "
                        "| valid loss {:5.2f}".format(
                            self.train_step // self.args.eval_interval,
                            self.train_step,
                            (time.time() - self.eval_start_time),
                            val_loss,
                        )
                    )
                    log_str += " | bpc {:9.5f}".format(val_loss / math.log(2))
                    log_str += " | valid ppl {:9.3f}".format(math.exp(val_loss))

                    if self.args.wandb:
                        self.wandb.log(
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
                        raw_model = model.module if hasattr(model, "module") else model
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
                    self.early_stop = True

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
                    self.wandb.log({"sample_accuracy": len(good_samples)})

            if self.train_step == self.args.max_step or self.early_stop:
                if self.early_stop and self.args.wandb:
                    self.wandb.run.summary["early_stop"] = self.train_step
                break
        if (
            self.early_stop
        ):
            self.logger("-" * 100)
            self.logger("early stopping of training")
            return
