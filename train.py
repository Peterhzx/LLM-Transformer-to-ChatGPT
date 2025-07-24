import json
import sys

from jsonschema import validate

from src.dataloader import Dataloader
from src.tokenizer import Tokenizer


def load_hyperparam(json_path=r".\config\config.json"):
    with open(json_path, 'r') as f:
        params = json.load(f)
    with open(r".\config\schema.json", 'r') as f:
        schema = json.load(f)
    validate(instance=params, schema=schema)
    return params


def train_tokenizer(params):
    dataloader = Dataloader(params["data_path"], params["allowed_chars"])
    tokenizer = Tokenizer()
    data_df = dataloader.get_df()
    tokenizer.train(data_df, params["vocab_size"])
    tokenizer.save()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)

    hyperparams = load_hyperparam(sys.argv[1])
    train_tokenizer(hyperparams)
