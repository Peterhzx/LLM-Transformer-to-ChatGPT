import os
import time
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from models.original_transformer import Transformer


class Trainer:
    def __init__(self, hyperparams):
        self._init_ckpt_dir(hyperparams)
        self.pad_token_id = hyperparams["model"]["params"]["pad_token_id"]
        self.num_epoch = hyperparams["num_epoch"]
        self.save_period = hyperparams["save_period"]["value"]
        self.device = self._check_cuda_availability()
        self._init_model(hyperparams["model"])
        self.model.apply(self._init_weights)
        self.model.to(self.device)
        self._init_optimizer(hyperparams["optimizer"])
        if "lr_scheduler" in hyperparams:
            self._init_lr_scheduler(hyperparams["lr_scheduler"])
        else:
            self.scheduler = None
        self._init_criterion(hyperparams["criterion"])

    def _init_ckpt_dir(self, hyperparams):
        if hyperparams["resume"]["value"]:
            self.ckpt_dir = hyperparams["resume"]["ckpt_dir"]
        else:
            now = "_".join([str(item) for item in time.localtime()[:6]])
            model_type = hyperparams["model"]["type"].lower()
            try:
                self.ckpt_dir = r"./checkpoints/"
                os.mkdir(self.ckpt_dir)
                self.ckpt_dir = r"./checkpoints/" + model_type + r"/"
                os.mkdir(self.ckpt_dir)
                self.ckpt_dir = r"./checkpoints/" + model_type + r"/" + now + "/"
                os.mkdir(self.ckpt_dir)
            except FileExistsError:
                try:
                    self.ckpt_dir = r"./checkpoints/" + model_type + r"/"
                    os.mkdir(self.ckpt_dir)
                    self.ckpt_dir = r"./checkpoints/" + model_type + r"/" + now + "/"
                    os.mkdir(self.ckpt_dir)
                except FileExistsError:
                    self.ckpt_dir = r"./checkpoints/" + model_type + r"/" + now + "/"
                    os.mkdir(self.ckpt_dir)

    def _init_model(self, hyperparams):
        if hyperparams["type"] == "Transformer":
            self.model = Transformer(**hyperparams["params"])
        else:
            raise ValueError("Invalid model type")

    def _init_optimizer(self, hyperparams):
        optim = getattr(torch.optim, hyperparams["type"])
        if hyperparams["type"] == "AdamW" or hyperparams["type"] == "Adam":
            params = hyperparams["params"].copy()
            params["betas"] = tuple(params["betas"])
            self.optimizer = optim(self.model.parameters(), **params)
        else:
            self.optimizer = optim(self.model.parameters(), **hyperparams["params"])

    def _init_lr_scheduler(self, hyperparams):
        lr_sch = getattr(torch.optim.lr_scheduler, hyperparams["type"])
        if hyperparams["type"] == "LambdaLR":
            if hyperparams["lambda"]["type"] == "Transformer_lambda":
                embed_dim = hyperparams["lambda"]["param"]["embed_dim"]
                warmup_steps = hyperparams["lambda"]["param"]["warmup_steps"]
                lambda_lr = lambda step: embed_dim ** -0.5 * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
                self.scheduler = lr_sch(self.optimizer, lr_lambda=lambda_lr)
            else:
                raise ValueError("Invalid lambda type")
        else:
            self.scheduler = lr_sch(self.optimizer, **hyperparams["param"])

    def _init_criterion(self, hyperparams):
        crit = getattr(nn, hyperparams["type"])
        self.criterion = crit(**hyperparams["params"])

    @staticmethod
    def _check_cuda_availability():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        return device

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _reset_from_last_checkpoint(self):
        ckpt_files = sorted(Path(self.ckpt_dir).glob("*.ckpt"), key=os.path.getmtime)
        if not ckpt_files:
            print("[Error] No checkpoint found to reset.")
            return 0, 0
        last_ckpt = ckpt_files[-1]
        print(f"[Reloading checkpoint] {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', self.scheduler.state_dict()))
        current_step = checkpoint.get('current_step', 0)
        current_epoch = checkpoint.get('current_epoch', 0)
        return current_step, current_epoch

    @staticmethod
    def _text_save(filename, data1):
        file = open(filename, 'a')
        for i in range(len(data1)):
            s1 = str(data1[i]).replace('[', '').replace(']', '')
            s1 = s1.replace("'", '').replace(',', '') + '\n'
            file.write(s1)
        file.close()

    def _train(self, epoch, current_step, train_loader, train_acc, train_loss_array):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        train_iter = iter(train_loader)
        train_iter = islice(train_iter, current_step, None)
        pbar = tqdm(train_iter, total=len(train_loader), desc="Training", initial=current_step)
        for batch_idx, (src, decoder_input, targets) in enumerate(pbar, start=current_step):
            src, decoder_input, targets = src.to(self.device), decoder_input.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            try:
                outputs = self.model(src, decoder_input)
                if torch.isnan(outputs).any():
                    raise ValueError("NaN in model output")
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = self.criterion(outputs, targets)
                if torch.isnan(loss):
                    raise ValueError("NaN in loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)  # It prevents exploding gradients in deep Transformer models.
                self.optimizer.step()
                self.scheduler.step()
            except ValueError as e:
                print(f"[Warning] {e} at step {batch_idx}. Reloading last checkpoint...")
                _, _ = self._reset_from_last_checkpoint()
                continue  # Skip this batch
            train_loss += loss.item()
            predicted = outputs.argmax(dim=-1)
            mask = (targets != self.pad_token_id)
            correct += ((predicted == targets) & mask).sum().float()
            total += mask.sum().float()
            if (batch_idx + 1) % self.save_period == 0:
                torch.save({
                    'current_epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'current_step': batch_idx + 1,
                }, self.ckpt_dir + f"epoch{epoch}_step{batch_idx + 1}.ckpt")

            pbar.set_postfix(loss=f"{train_loss / (batch_idx + 1):.4f}", acc=f"{correct / total:.2%}")
            train_acc.append(correct / total)
            train_loss_array.append(train_loss / len(train_loader))
        torch.save({
            'current_epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, self.ckpt_dir + f"epoch{epoch + 1}.ckpt")

    def _val(self, val_loader, valid_acc, valid_loss_array):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
            for batch_idx, (src, decoder_input, targets) in pbar:
                src, decoder_input, targets = src.to(self.device), decoder_input.to(self.device), targets.to(self.device)
                try:
                    outputs = self.model(src, decoder_input)
                    if torch.isnan(outputs).any():
                        raise ValueError("NaN in model output")
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                    loss = self.criterion(outputs, targets)
                    if torch.isnan(loss).any():
                        raise ValueError("NaN in loss")
                except ValueError as err:
                    print(f"[Warning] {err} at step {batch_idx}.")
                    continue  # Skip this batch
                test_loss += loss.item()
                predicted = outputs.argmax(dim=-1)
                correct += (predicted == targets).sum().item()
                total += targets.numel()

                pbar.set_postfix(loss=f"{test_loss / (batch_idx + 1):.4f}", acc=f"{correct / total:.2%}")
        valid_acc.append(correct / total)
        valid_loss_array.append(test_loss / len(val_loader))

    def train(self, train_loader, val_loader):
        current_step, current_epoch = self._reset_from_last_checkpoint()
        train_acc = []
        valid_acc = []
        train_loss_array = []
        valid_loss_array = []
        for epoch in range(current_epoch, self.num_epoch):
            self._train(epoch, current_step, train_loader, train_acc, train_loss_array)
            self._val(val_loader, valid_acc, valid_loss_array)
            current_step = 0

        self._text_save(self.ckpt_dir + "train_acc.txt", train_acc)
        self._text_save(self.ckpt_dir + "valid_acc.txt", valid_acc)
        self._text_save(self.ckpt_dir + "train_loss.txt", train_loss_array)
        self._text_save(self.ckpt_dir + "valid_loss.txt", valid_loss_array)

    def save(self):
        torch.save(self.model.state_dict(), self.ckpt_dir + "weights.pt")
        """
        src_tensor, decoder_input_tensor, _ = train_set.__getitem__(0)
        torch.onnx.export(
            self.model,  # model to export
            (src_tensor, decoder_input_tensor),  # inputs of the model,
            "my_model.onnx",  # filename of the ONNX model
            input_names=["input"],  # Rename inputs for the ONNX model
            dynamo=True  # True or False to select the exporter to use
        )
        """
