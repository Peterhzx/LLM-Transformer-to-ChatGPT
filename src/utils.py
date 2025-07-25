import json
import sys

import jsonschema
from jsonschema import validate

from src.dataloader import Dataloader
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
    data_df = dataloader.get_df(params["sample_size"])
    tokenizer.train(data_df, params)
    return tokenizer


def save_tokenizer(tokenizer, path=None):
    tokenizer.save(path)


def load_tokenizer(tokenizer, path=None):
    tokenizer.load(path)


def train_model(dataloader, params):
    loader = getattr(dataloader, params["dataloader"]["type"])
    train_loader, val_loader, _ = loader(**params["dataloader"]["params"])
    trainer = Trainer(params)
    trainer.train(train_loader, val_loader)
    trainer.save()

# def eval_model(model, dataloader, params):
