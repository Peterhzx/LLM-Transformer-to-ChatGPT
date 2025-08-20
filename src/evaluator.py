import json
import os
from pathlib import Path

import torch
from torchmetrics.text import BLEUScore
from tqdm import tqdm

import models


class Evaluator:
    def __init__(self, params, tokenizer=None, trainer=None):
        if trainer is not None and tokenizer is not None:
            self.device = trainer.get_device()
            self.model = trainer.get_model()
            self.ckpt_dir = trainer.get_weights_dir()
            self.reversed_tokens = tokenizer.reversed_tokens
        else:
            self._check_cuda_availability()
            self._init_model(params["model"])
            self._load_from_last_weights(params["ckpt_dir"])
            self._load_reversed_tokens(params["reversed_tokens_dir"])

    def _load_reversed_tokens(self, path):
        with open(path + r"reversed_tokens.json", "r") as f:
            self.reversed_tokens = json.load(f)
        print(f"reversed_tokens loaded from {path}")

    def _check_cuda_availability(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

    def _init_model(self, hyperparams):
        model_type = getattr(models, hyperparams["type"])
        self.model = model_type(**hyperparams["params"])
        self.model.to(self.device)

    def _load_from_last_weights(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        pt_files = sorted(Path(ckpt_dir).glob("*.pt"), key=os.path.getmtime)
        if not pt_files:
            print("[Error] No weights found to load.")
        else:
            last_pt = pt_files[-1]
            print(f"[Loading checkpoint] {last_pt}")
            weights = torch.load(last_pt, map_location=self.device)
            self.model.load_state_dict(weights['model_state_dict'])

    def _prep_for_eval(self, tensor):
        return [[self.reversed_tokens.get(str(item), item) for item in sentence if item not in {0, 1, 2}] for sentence in tensor]

    def evaluate(self, test_loader):
        bleu_metric = BLEUScore(n_gram=4, smooth=True).to(self.device)
        bleu_metric.reset()
        self.model.eval()

        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        with torch.no_grad():
            for batch_idx, (src, decoder_input, targets) in pbar:
                src, decoder_input = src.to(self.device), decoder_input.to(self.device)
                outputs = self.model(src, decoder_input)
                predicted = outputs.argmax(dim=-1).cpu().tolist()
                predicted = self._prep_for_eval(predicted)
                targets = self._prep_for_eval(targets.tolist())
                predicted = [' '.join(tokens) for tokens in predicted]
                targets = [[' '.join(tokens)] for tokens in targets]
                bleu_metric.update(predicted, targets)

        final_bleu = bleu_metric.compute()
        print(f"Final BLEU: {final_bleu.item():.4f}")
        with open(self.ckpt_dir + "bleu_score.txt", "a") as f:
            f.write(f"Final BLEU: {final_bleu.item():.4f}")
