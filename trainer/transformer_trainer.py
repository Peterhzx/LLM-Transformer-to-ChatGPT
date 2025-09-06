import time
from itertools import islice

import torch
from tqdm import tqdm

from trainer import Trainer


class TransformerTrainer(Trainer):
    def __init__(self, hyperparams, num_tokens):
        super(TransformerTrainer, self).__init__()
        self.pad_token_id = hyperparams["model"]["params"]["pad_token_id"]
        self.num_epoch = hyperparams["num_epoch"]
        self.save_period = hyperparams["save_period"]["value"]
        self._check_cuda_availability()
        self._init_ckpt_dir(hyperparams)
        self._init_model(hyperparams["model"], num_tokens)
        self._init_optimizer(hyperparams["optimizer"])
        if "lr_scheduler" in hyperparams:
            self._init_lr_scheduler(hyperparams["lr_scheduler"])
        else:
            self.scheduler = None
        self._init_criterion(hyperparams["criterion"])

    def _train(self, epoch, current_step, train_loader, train_acc, train_loss_array):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        # torch.autograd.set_detect_anomaly(True)
        time.sleep(0.5)
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
            pbar.set_postfix(loss=f"{train_loss / num_batch:.4f}", acc=f"{correct / total:.2%}")
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
