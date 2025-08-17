import json
import sys

import jsonschema
from jsonschema import validate

from src.dataloader import Dataloader
from src.evaluator import Evaluator
from src.trainer import Trainer
from tokenizer.bpe import BPE


def load_config(json_path=r".\config\config.json"):
    with open(json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    with open(r".\config\schema.json", 'r') as f:
        schema = json.load(f)
    try:
        validate(instance=params, schema=schema)
        print("Valid configuration!")
        return params
    except jsonschema.exceptions.ValidationError as err:
        print(f"Invalid configuration: {err}")
        sys.exit(1)


def load_data(params):
    dataloader = Dataloader(params)
    return dataloader


def train_tokenizer(dataloader, params):
    tokenizer = BPE()
    if params["load"]["value"]:
        tokenizer.load(params["load"]["dir"])
    else:
        data_df = dataloader.get_df(params["sample_size"])
        tokenizer.train(data_df, params)
    return tokenizer


def save_tokenizer(tokenizer, path=None):
    tokenizer.save(path)


def tokenize_data(dataloader, tokenizer, params):
    dataloader.tokenize_df(tokenizer, params)


def train_and_save_model(dataloader, params):
    loader = getattr(dataloader, params["dataloader"]["type"])
    train_loader, val_loader, test_loader = loader(**params["dataloader"]["params"])
    trainer = Trainer(params)
    trainer.train(train_loader, val_loader)
    trainer.save()
    return trainer, test_loader


def eval_model(params, test_loader, tokenizer=None, trainer=None):
    evaluator = Evaluator(params, tokenizer, trainer)
    evaluator.evaluate(test_loader)
