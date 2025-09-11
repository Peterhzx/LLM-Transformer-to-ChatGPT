import json
import logging
import os
import time
from itertools import islice

import torch
from torch import nn
from torch import amp
from tqdm import tqdm

from trainer import Trainer


class BERTTrainer(Trainer):
    def __init__(self, num_epoch, save_period, resume, model, optimizer, lr_scheduler, criterion, num_tokens, mode, max_num_ckpt=-1, enable_amp=True, **kwargs):
        super(BERTTrainer, self).__init__()
        self.pad_token_id = model["params"]["pad_token_id"]
        self.num_epoch = num_epoch
        self.save_period = save_period
        self.max_num_ckpt = max_num_ckpt
        self.enable_amp = enable_amp
        self.mode = mode
        self._init_dir(resume, model, mode)
        self.cuda_availability = self._check_cuda_availability()
        self._init_model(model, num_tokens)
        self._init_optimizer(optimizer)
        self._init_lr_scheduler(lr_scheduler)
        self._init_criterion(criterion)
        self.criterion_nsp = nn.CrossEntropyLoss()
        self.scaler = amp.GradScaler("cuda", enabled=self.enable_amp) if self.cuda_availability else None

    def _save_acc_loss(self, train_loss, mlm_train_loss, nsp_train_loss, nsp_correct, nsp_total, correct, total):
        resume_acc_loss = {
            "train_loss": train_loss,
            "mlm_train_loss": mlm_train_loss,
            "nsp_train_loss": nsp_train_loss,
            "nsp_correct": nsp_correct,
            "nsp_total": nsp_total,
            "correct": correct,
            "total": total
        }
        with open(os.path.join(self.ckpt_dir, "resume_acc_loss.json"), "w") as f:
            json.dump(resume_acc_loss, f, indent=4)

    def _train(self, epoch, current_step, train_loader, resume_acc_loss):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        time.sleep(0.5)
        train_acc = []
        train_loss_array = []
        train_loss = resume_acc_loss.get("train_loss", 0)
        mlm_train_loss = resume_acc_loss.get("mlm_train_loss", 0)
        nsp_train_loss = resume_acc_loss.get("nsp_train_loss", 0)
        nsp_correct = resume_acc_loss.get("nsp_correct", 0)
        nsp_total = resume_acc_loss.get("nsp_total", 0)
        correct = resume_acc_loss.get("correct", 0)
        total = resume_acc_loss.get("total", 0)
        train_iter = iter(train_loader)
        train_iter = islice(train_iter, current_step, None)
        pbar = tqdm(train_iter, total=len(train_loader), desc="Training", initial=current_step)
        for batch_idx, (src, targets, mask_tensor) in enumerate(pbar, start=current_step):
            src, targets, mask_tensor = src.to(self.device), targets.to(self.device), mask_tensor.to(self.device)
            self.optimizer.zero_grad()
            with amp.autocast("cuda", enabled=self.enable_amp and self.cuda_availability):
                mlm_logits, nsp_logits = self.model(src)

                masked_logits = mlm_logits.reshape(-1, mlm_logits.size(-1))
                masked_targets = targets.reshape(-1)
                mask_tensor = mask_tensor.reshape(-1)
                masked_logits = masked_logits[mask_tensor, :]
                masked_targets = masked_targets[mask_tensor]

                mlm_loss = self.criterion(masked_logits, masked_targets)
                predicted = masked_logits.argmax(dim=-1)
                correct += (predicted == masked_targets).sum().float().item()
                total += float(len(masked_targets))

                nsp_loss = self.criterion_nsp(nsp_logits, targets[:, 0])
                nsp_predicted = nsp_logits.argmax(dim=-1).view(-1)
                nsp_targets = targets[:, 0].view(-1)
                nsp_correct += (nsp_predicted == nsp_targets).sum().float().item()
                nsp_total += float(len(nsp_targets))

                loss = mlm_loss + nsp_loss

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
            train_loss += loss.item()
            mlm_train_loss += mlm_loss.item()
            nsp_train_loss += nsp_loss.item()

            pbar.set_postfix(loss=f"{train_loss / (batch_idx + 1):.4f}", mlm_loss=f"{mlm_train_loss / (batch_idx + 1):.4f}",
                             nsp_loss=f"{nsp_train_loss / (batch_idx + 1):.4f}", acc=f"{correct / total:.2%}",
                             nsp_acc=f"{nsp_correct / nsp_total:.2%}")
            train_acc.append(correct / total)
            train_loss_array.append(train_loss / (batch_idx + 1))

            if (batch_idx + 1) % self.save_period == 0:
                checkpoint = {
                    'current_epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'current_step': batch_idx + 1,
                }
                if self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                if self.scaler:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()

                if self.max_num_ckpt == -1:
                    torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch}_step{batch_idx + 1}.ckpt"))
                elif self.max_num_ckpt > 0:
                    torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch}_step{batch_idx + 1}.ckpt"))
                    self._remove_ckpt_exceeding_limit(self.max_num_ckpt)
                self._save_acc_loss(train_loss, mlm_train_loss, nsp_train_loss, nsp_correct, nsp_total, correct, total)

                self._text_save(os.path.join(self.output_dir, f"train_acc_epoch_{epoch}.txt"), train_acc)
                self._text_save(os.path.join(self.output_dir, f"train_loss_epoch_{epoch}.txt"), train_loss_array)
                logging.info(f"loss={train_loss / (batch_idx + 1):.4f}, mlm_loss={mlm_train_loss / (batch_idx + 1):.4f}, nsp_loss={nsp_train_loss / (batch_idx + 1):.4f}, acc={correct / total:.2%}, nsp_acc={nsp_correct / nsp_total:.2%} at epoch {epoch}, step {batch_idx + 1}")
                train_acc = []
                train_loss_array = []

        checkpoint = {
            'current_epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.max_num_ckpt == -1:
            torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch + 1}.ckpt"))
        elif self.max_num_ckpt > 0:
            torch.save(checkpoint, os.path.join(self.ckpt_dir, f"epoch{epoch + 1}.ckpt"))
            self._remove_ckpt_exceeding_limit(self.max_num_ckpt)
        self._save_acc_loss(0, 0, 0, 0, 0, 0, 0)

        self._text_save(os.path.join(self.output_dir, f"train_acc_epoch_{epoch}.txt"), train_acc)
        self._text_save(os.path.join(self.output_dir, f"train_loss_epoch_{epoch}.txt"), train_loss_array)
        logging.info(f"loss={train_loss / len(train_loader):.4f}, mlm_loss={mlm_train_loss / len(train_loader):.4f}, nsp_loss={nsp_train_loss / len(train_loader):.4f}, acc={correct / total:.2%}, nsp_acc={nsp_correct / nsp_total:.2%} at epoch {epoch}")

    def _val(self, epoch, val_loader):
        self.model.eval()
        valid_acc = []
        valid_loss_array = []
        test_loss = 0
        mlm_test_loss = 0
        nsp_test_loss = 0
        nsp_correct = 0
        nsp_total = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
            for batch_idx, (src, targets, mask_tensor) in pbar:
                src, targets, mask_tensor = src.to(self.device), targets.to(self.device), mask_tensor.to(self.device)
                with amp.autocast("cuda", enabled=self.enable_amp and self.cuda_availability):
                    mlm_logits, nsp_logits = self.model(src)

                    masked_logits = mlm_logits.reshape(-1, mlm_logits.size(-1))
                    masked_targets = targets.reshape(-1)
                    mask_tensor = mask_tensor.reshape(-1)
                    masked_logits = masked_logits[mask_tensor, :]
                    masked_targets = masked_targets[mask_tensor]

                    mlm_loss = self.criterion(masked_logits, masked_targets)
                    predicted = masked_logits.argmax(dim=-1)
                    correct += (predicted == masked_targets).sum().float().item()
                    total += float(len(masked_targets))

                    nsp_loss = self.criterion_nsp(nsp_logits, targets[:, 0])
                    nsp_predicted = nsp_logits.argmax(dim=-1).view(-1)
                    nsp_targets = targets[:, 0].view(-1)
                    nsp_correct += (nsp_predicted == nsp_targets).sum().float().item()
                    nsp_total += float(len(nsp_targets))

                    loss = mlm_loss + nsp_loss

                test_loss += loss.item()
                mlm_test_loss += mlm_loss.item()
                nsp_test_loss += nsp_loss.item()

                pbar.set_postfix(loss=f"{test_loss / (batch_idx + 1):.4f}", mlm_loss=f"{mlm_test_loss / (batch_idx + 1):.4f}", nsp_loss=f"{nsp_test_loss / (batch_idx + 1):.4f}", acc=f"{correct / total:.2%}", nsp_acc=f"{nsp_correct / nsp_total:.2%}")
                valid_acc.append(correct / total)
                valid_loss_array.append(test_loss / (batch_idx + 1))
        self._text_save(os.path.join(self.output_dir, f"valid_acc_epoch_{epoch}.txt"), valid_acc)
        self._text_save(os.path.join(self.output_dir, f"valid_loss_epoch_{epoch}.txt"), valid_loss_array)
        logging.info(f"loss={test_loss / len(val_loader):.4f}, mlm_loss={mlm_test_loss / len(val_loader):.4f}, nsp_loss={nsp_test_loss / len(val_loader):.4f}, acc={correct / total:.2%}, nsp_acc={nsp_correct / nsp_total:.2%} at epoch {epoch} in val")
