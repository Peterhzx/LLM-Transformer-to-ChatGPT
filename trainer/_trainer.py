import logging
import os
import sys
import time
from abc import ABC
from pathlib import Path

import torch
from torch import nn

import models


class Trainer(ABC):
    def __init__(self):
        super(ABC, self).__init__()
        self.mode = None
        self.pad_token_id = None
        self.num_epoch = None
        self.save_period = None

    def _init_ckpt_dir(self, hyperparams, mode):
        if mode == "sagemaker":
            parent = "/opt/ml/checkpoints"
        elif mode == "local":
            parent = "./checkpoints"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        try:
            if hyperparams["resume"]["value"]:
                if "ckpt_dir" in hyperparams["resume"]:
                    self.ckpt_dir = hyperparams["resume"]["ckpt_dir"]
                else:
                    model_type = hyperparams["model"]["type"].lower()
                    base_dir = Path(parent) / model_type

                    # Check if base directory exists
                    if not base_dir.exists():
                        print(f"[Error] Checkpoint directory {base_dir} does not exist.")
                        sys.exit(-1)

                    # Get all subdirectories and sort by modification time
                    dirs = [d for d in base_dir.iterdir() if d.is_dir()]
                    if not dirs:
                        print("[Error] No checkpoint found to resume.")
                        sys.exit(-1)

                    dirs.sort(key=os.path.getmtime)
                    self.ckpt_dir = str(dirs[-1])
            else:
                self.ckpt_dir = hyperparams["resume"]["ckpt_dir"]
        except KeyError:
            now = "_".join([str(item) for item in time.localtime()[:6]])
            model_type = hyperparams["model"]["type"].lower()
            self.ckpt_dir = os.path.join(parent, model_type, now)
            os.makedirs(self.ckpt_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.ckpt_dir, 'TRAINING.log'),
            level=logging.INFO,
            format='%(asctime)s - Trainer - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _init_model(self, hyperparams, num_tokens):
        model_name = hyperparams["type"]
        model_type = getattr(models, model_name)
        self.model = model_type(num_tokens=num_tokens, **hyperparams["params"])
        self.model.apply(self._init_weights)
        self.model.to(self.device)
        logging.info(f"Training {model_name} model")

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
                embed_dim = hyperparams["lambda"]["params"]["embed_dim"]
                warmup_steps = hyperparams["lambda"]["params"]["warmup_steps"]
                lambda_lr = lambda step: embed_dim ** -0.5 * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
                self.scheduler = lr_sch(self.optimizer, lr_lambda=lambda_lr)
            else:
                raise ValueError("Invalid lambda type")
        else:
            self.scheduler = lr_sch(self.optimizer, **hyperparams["param"])

    def _init_criterion(self, hyperparams):
        crit = getattr(nn, hyperparams["type"])
        self.criterion = crit(**hyperparams["params"])

    def _check_cuda_availability(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

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
            print("[Warning] No checkpoint found to reset.")
            logging.warning("No checkpoint found to reset.")
            return 0, 0
        last_ckpt = ckpt_files[-1]
        print(f"[Reloading checkpoint] {last_ckpt}")
        logging.info(f"Reloading checkpoint {last_ckpt}.")
        checkpoint = torch.load(last_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', self.scheduler.state_dict()))
        current_step = checkpoint.get('current_step', 0)
        current_epoch = checkpoint.get('current_epoch', 0)
        return current_step, current_epoch

    def _remove_ckpt_exceeding_limit(self, limit=2):
        ckpt_files = sorted(Path(self.ckpt_dir).glob("*.ckpt"), key=os.path.getmtime)
        if len(ckpt_files) <= limit or not ckpt_files:
            logging.warning("No checkpoint to be removed.")
            return
        else:
            for file_path in ckpt_files[:-limit]:
                try:
                    file_path.unlink()
                    logging.info(f"Over limited checkpoint removed: {file_path}")
                except Exception as e:
                    logging.error(f"Unable to remove {file_path}: {e}")

    @staticmethod
    def _text_save(filename, data1):
        file = open(filename, 'a')
        for i in range(len(data1)):
            s1 = str(data1[i]).replace('[', '').replace(']', '')
            s1 = s1.replace("'", '').replace(',', '') + '\n'
            file.write(s1)
        file.close()

    def _train(self, epoch, current_step, train_loader):
        pass

    def _val(self, epoch, val_loader):
        pass

    def train(self, train_loader, val_loader):
        current_step, current_epoch = self._reset_from_last_checkpoint()
        for epoch in range(current_epoch, self.num_epoch):
            self._train(epoch, current_step, train_loader)
            self._val(epoch, val_loader)
            current_step = 0

    def save(self):
        if self.mode == "sagemaker":
            parent = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output/data")
            file_dir = os.path.join(parent, "weights.pt")
            torch.save(self.model.state_dict(), file_dir)
            print(f"model saved to {file_dir}")
            logging.info(f"model saved to {file_dir}")
        elif self.mode == "local":
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "weights.pt"))
            print("model saved to " + self.ckpt_dir + "/weights.pt")
            logging.info(f"model saved to {self.ckpt_dir}/weights.pt")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

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

    def get_weights_dir(self):
        return self.ckpt_dir

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device
