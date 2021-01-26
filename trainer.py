import itertools
import math
import os
import time

import torch
import wandb


from utils.sample import sample_words


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_iter,
        eval_iter,
        test_iter,
        logger,
        corpus,
        args,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.test_iter = test_iter
        self.args = args
        self.early_stop = False
        self.train_step = 0
        self.train_loss = 0
        self.best_val_loss = None
        self.early_stop = False
        self.logger = logger
        self.corpus = corpus
        self.log_start_time = time.time()
        self.eval_start_time = time.time()
        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def evaluate(self):
        # Turn on evaluation mode which disables dropout.

        self.model.eval()

        # Evaluation
        total_len, total_loss = 0, 0.0
        with torch.no_grad():
            for i, (data, target, seq_len) in enumerate(self.eval_iter):
                if self.args.max_eval_steps > 0 and i >= self.args.max_eval_steps:
                    break
                logits, loss = self.model(data, target)
                loss = loss.mean()
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

        # Switch back to the training mode
        self.model.train()

        return total_loss / total_len, total_len

    def test(self, model):
        # Turn on evaluation mode which disables dropout.

        model.eval()

        # Evaluation
        total_len, total_loss = 0, 0.0
        with torch.no_grad():
            for i, (data, target, seq_len) in enumerate(self.test_iter):
                if self.args.max_eval_steps > 0 and i >= self.args.max_eval_steps:
                    break
                logits, loss = model(data, target)
                loss = loss.mean()
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

        return total_loss / total_len, total_len

    def train(self):
        # Turn on training mode which enables dropout.
        self.model.train()
        for epoch in itertools.count(start=1):
            for batch, (data, target, seq_len) in enumerate(self.train_iter):
                logits, loss = self.model(data, target)
                self.model.zero_grad()

                loss.backward()
                self.train_loss += loss.float().item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                self.optimizer.step()

                # step-wise learning rate annealing
                self.train_step += 1
                if self.args.scheduler in [
                    "cosine",
                    "constant",
                ]:
                    # linear warmup stage
                    if self.train_step < self.args.warmup_step:
                        curr_lr = self.args.lr * self.train_step / self.args.warmup_step
                        self.optimizer.param_groups[0]["lr"] = curr_lr
                    else:
                        if self.args.scheduler == "cosine":
                            self.scheduler.step()
                elif self.args.scheduler == "inv_sqrt":
                    self.scheduler.step()

                if self.train_step % self.args.log_interval == 0:
                    cur_loss = self.train_loss / self.args.log_interval
                    elapsed = time.time() - self.log_start_time
                    log_str = (
                        "| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} "
                        "| ms/batch {:5.2f} | loss {:5.2f}".format(
                            epoch,
                            self.train_step,
                            batch + 1,
                            self.optimizer.param_groups[0]["lr"],
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
                                "tokens": (batch + 1) * self.args.n_ctx,
                                "lr": self.optimizer.param_groups[0]["lr"],
                            },
                        )
                    self.train_loss = 0
                    self.log_start_time = time.time()

                if self.train_step % self.args.eval_interval == 0:
                    val_loss, total_tokens = self.evaluate()
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
                        if not self.args.debug:
                            self.logger("saving new model")
                            with open(
                                os.path.join(self.args.work_dir, "model.pt"), "wb"
                            ) as f:

                                torch.save(self.model, f)
                            with open(
                                os.path.join(self.args.work_dir, "optimizer.pt"), "wb"
                            ) as f:
                                torch.save(self.optimizer.state_dict(), f)
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
                ):
                    # sample here
                    words = sample_words(
                        self.model,
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
            if (
                (self.args.max_epoch and self.args.max_epoch == epoch)
                or self.train_step == self.args.max_step
                or self.early_stop
            ):
                self.logger("-" * 100)
                self.logger("End of training")
                break
