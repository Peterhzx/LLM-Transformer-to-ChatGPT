import logging
import os
import time
from itertools import islice

import torch
from tqdm import tqdm

from trainer import Trainer


class TransformerTrainer(Trainer):
    def __init__(self, hyperparams, num_tokens, mode):
        super(TransformerTrainer, self).__init__()
        self.pad_token_id = hyperparams["model"]["params"]["pad_token_id"]
        self.num_epoch = hyperparams["num_epoch"]
        self.save_period = hyperparams["save_period"]["value"]
        self.max_num_ckpt = hyperparams.get("max_num_ckpt", -1)
        self.mode = mode
        self._check_cuda_availability()
        self._init_ckpt_dir(hyperparams, mode)
        self._init_model(hyperparams["model"], num_tokens)
        self._init_optimizer(hyperparams["optimizer"])
        if "lr_scheduler" in hyperparams:
            self._init_lr_scheduler(hyperparams["lr_scheduler"])
        else:
            self.scheduler = None
        self._init_criterion(hyperparams["criterion"])

    def _train(self, epoch, current_step, train_loader):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        time.sleep(0.5)
        train_acc = []
        train_loss_array = []
        train_loss = 0
        num_batch = 0
        correct = 0
        total = 0
        train_iter = iter(train_loader)
        train_iter = islice(train_iter, current_step, None)
        pbar = tqdm(train_iter, total=len(train_loader), desc="Training", initial=current_step)
        for batch_idx, (src, decoder_input, targets) in enumerate(pbar, start=current_step):
            src, decoder_input, targets = src.to(self.device), decoder_input.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(src, decoder_input)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # It prevents exploding gradients in deep Transformer models.
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            train_loss += loss.item()
            predicted = outputs.argmax(dim=-1)
            mask = (targets != self.pad_token_id)
            correct += ((predicted == targets) & mask).sum().float()
            total += mask.sum().float()

            num_batch += 1
            pbar.set_postfix(loss=f"{train_loss / num_batch:.4f}", acc=f"{correct / total:.2%}")
            train_acc.append(correct / total)
            train_loss_array.append(train_loss / num_batch)

            if (batch_idx + 1) % self.save_period == 0:
                checkpoint = {
                    'current_epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'current_step': batch_idx + 1,
                }
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

                if self.max_num_ckpt == -1:
                    torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch}_step{batch_idx + 1}.ckpt"))
                elif self.max_num_ckpt > 0:
                    torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch}_step{batch_idx + 1}.ckpt"))
                    self._remove_ckpt_exceeding_limit(self.max_num_ckpt)

                self._text_save(os.path.join(self.ckpt_dir, f"train_acc_epoch_{epoch}.txt"), train_acc)
                self._text_save(os.path.join(self.ckpt_dir, f"train_loss_epoch_{epoch}.txt"), train_loss_array)
                logging.info(f"loss={train_loss / num_batch:.4f}, acc={correct / total:.2%} at epoch {epoch}, step {batch_idx + 1}")
                train_acc = []
                train_loss_array = []

        checkpoint = {
            'current_epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.max_num_ckpt == -1:
            torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch + 1}.ckpt"))
        elif self.max_num_ckpt > 0:
            torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch + 1}.ckpt"))
            self._remove_ckpt_exceeding_limit(self.max_num_ckpt)

        self._text_save(os.path.join(self.ckpt_dir, f"train_acc_epoch_{epoch}.txt"), train_acc)
        self._text_save(os.path.join(self.ckpt_dir, f"train_loss_epoch_{epoch}.txt"), train_loss_array)
        logging.info(f"loss={train_loss / num_batch:.4f}, acc={correct / total:.2%} at epoch {epoch}")

    def _val(self, epoch, val_loader):
        self.model.eval()
        valid_acc = []
        valid_loss_array = []
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
            for batch_idx, (src, decoder_input, targets) in pbar:
                src, decoder_input, targets = src.to(self.device), decoder_input.to(self.device), targets.to(self.device)
                outputs = self.model(src, decoder_input)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                predicted = outputs.argmax(dim=-1)
                correct += (predicted == targets).sum().item()
                total += targets.numel()

                pbar.set_postfix(loss=f"{test_loss / (batch_idx + 1):.4f}", acc=f"{correct / total:.2%}")
                valid_acc.append(correct / total)
                valid_loss_array.append(test_loss / len(val_loader))
        self._text_save(os.path.join(self.ckpt_dir, f"valid_acc_epoch_{epoch}.txt"), valid_acc)
        self._text_save(os.path.join(self.ckpt_dir, f"valid_loss_epoch_{epoch}.txt"), valid_loss_array)
        logging.info(f"loss={test_loss / (batch_idx + 1):.4f}, acc={correct / total:.2%} at epoch {epoch}")
