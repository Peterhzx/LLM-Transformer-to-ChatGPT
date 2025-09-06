import time
from itertools import islice

import torch
from torch import nn
from tqdm import tqdm

from trainer import Trainer


class BERTTrainer(Trainer):
    def __init__(self, hyperparams, num_tokens):
        super(BERTTrainer, self).__init__()
        self.pad_token_id = hyperparams["model"]["params"]["pad_token_id"]
        self.num_epoch = hyperparams["num_epoch"]
        self.save_period = hyperparams["save_period"]["value"]
        self.batch_size = hyperparams["dataloader"]["params"]["batch_size"]
        self._check_cuda_availability()
        self._init_ckpt_dir(hyperparams)
        self._init_model(hyperparams["model"], num_tokens)
        self._init_optimizer(hyperparams["optimizer"])
        if "lr_scheduler" in hyperparams:
            self._init_lr_scheduler(hyperparams["lr_scheduler"])
        else:
            self.scheduler = None
        self._init_criterion(hyperparams["criterion"])
        self.criterion_nsp = nn.CrossEntropyLoss()

    def _train(self, epoch, current_step, train_loader, train_acc, train_loss_array):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        # torch.autograd.set_detect_anomaly(True)
        time.sleep(0.5)
        train_loss = 0
        mlm_train_loss = 0
        nsp_train_loss = 0
        num_batch = 0
        nsp_correct = 0
        nsp_total = 0
        correct = 0
        total = 0
        train_iter = iter(train_loader)
        train_iter = islice(train_iter, current_step, None)
        pbar = tqdm(train_iter, total=len(train_loader), desc="Training", initial=current_step)
        for batch_idx, (src, targets, pred_tensor) in enumerate(pbar, start=current_step):
            src, targets = src.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            mlm_logits, nsp_logits = self.model(src)

            mlm_loss = torch.tensor(0.0, device=self.device)
            pred_list = pred_tensor.tolist()
            for b in range(src.size(0)):
                masked_idx = pred_list[b]
                masked_idx = [int(x) for x in masked_idx if int(x) != self.pad_token_id]
                masked_logits = mlm_logits[b, masked_idx, :]
                masked_targets = targets[b, masked_idx]
                mlm_loss += self.criterion(masked_logits, masked_targets)
                predicted = masked_logits.argmax(dim=-1)
                correct += (predicted == masked_targets).sum().float()
                total += float(len(masked_targets))

            mlm_loss = mlm_loss / self.batch_size

            nsp_predicted = nsp_logits.argmax(dim=-1).view(-1)
            nsp_targets = targets[:, 0].view(-1)
            nsp_correct += (nsp_predicted == nsp_targets).sum().float()
            nsp_total += float(len(nsp_targets))

            nsp_loss = self.criterion_nsp(nsp_logits, targets[:, 0])

            loss = mlm_loss + nsp_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # It prevents exploding gradients in deep Transformer models.
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            train_loss += loss.item()
            mlm_train_loss += mlm_loss.item()
            nsp_train_loss += nsp_loss.item()
            if (batch_idx + 1) % self.save_period == 0:
                checkpoint = {
                    'current_epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'current_step': batch_idx + 1,
                }
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

                torch.save(checkpoint, self.ckpt_dir + f"epoch{epoch}_step{batch_idx + 1}.ckpt")

            num_batch += 1
            pbar.set_postfix(loss=f"{train_loss / num_batch:.4f}", mlm_loss=f"{mlm_train_loss / num_batch:.4f}", nsp_loss=f"{nsp_train_loss / num_batch:.4f}", acc=f"{correct / total:.2%}", nsp_acc=f"{nsp_correct / nsp_total:.2%}")
            train_acc.append(correct / total)
            train_loss_array.append(train_loss / num_batch)

        checkpoint = {
            'current_epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.ckpt_dir + f"epoch{epoch + 1}.ckpt")

    def _val(self, val_loader, valid_acc, valid_loss_array):
        self.model.eval()
        test_loss = 0
        mlm_test_loss = 0
        nsp_test_loss = 0
        num_batch = 0
        nsp_correct = 0
        nsp_total = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
            for batch_idx, (src, targets, pred_tensor) in pbar:
                src, targets = src.to(self.device), targets.to(self.device)
                mlm_logits, nsp_logits = self.model(src)

                mlm_loss = torch.tensor(0.0, device=self.device)
                pred_list = pred_tensor.tolist()
                for b in range(src.size(0)):
                    masked_idx = pred_list[b]
                    masked_idx = [int(x) for x in masked_idx if int(x) != self.pad_token_id]
                    masked_logits = mlm_logits[b, masked_idx, :]
                    masked_targets = targets[b, masked_idx]
                    mlm_loss += self.criterion(masked_logits, masked_targets)
                    predicted = masked_logits.argmax(dim=-1)
                    correct += (predicted == masked_targets).sum().float()
                    total += float(len(masked_targets))

                mlm_loss = mlm_loss / self.batch_size

                nsp_predicted = nsp_logits.argmax(dim=-1).view(-1)
                nsp_targets = targets[:, 0].view(-1)
                nsp_correct += (nsp_predicted == nsp_targets).sum().float()
                nsp_total += float(len(nsp_targets))

                nsp_loss = self.criterion_nsp(nsp_logits, targets[:, 0])

                loss = mlm_loss + nsp_loss
                test_loss += loss.item()
                mlm_test_loss += mlm_loss.item()
                nsp_test_loss += nsp_loss.item()

                num_batch += 1
                pbar.set_postfix(loss=f"{test_loss / num_batch:.4f}", mlm_loss=f"{mlm_test_loss / num_batch:.4f}", nsp_loss=f"{nsp_test_loss / num_batch:.4f}", acc=f"{correct / total:.2%}", nsp_acc=f"{nsp_correct / nsp_total:.2%}")
                valid_acc.append(correct / total)
                valid_loss_array.append(test_loss / num_batch)
