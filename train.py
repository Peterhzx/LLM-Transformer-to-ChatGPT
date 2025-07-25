import json
import sys

import jsonschema
from jsonschema import validate

from src.dataloader import Dataloader
from src.tokenizer import Tokenizer


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


def train_tokenizer(params):
    dataloader = Dataloader(params["Dataloader"])
    tokenizer = Tokenizer()
    data_df = dataloader.get_df(params["Tokenizer"]["sample_size"])
    tokenizer.train(data_df, params["Tokenizer"])
    tokenizer.save()


if __name__ == '__main__':  # .\\config\\config.json
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    config = load_config(sys.argv[1])
    train_tokenizer(config)
