import json

from jsonschema import validate

from src.dataloader import Dataloader
from src.tokenizer import Tokenizer


def load_hyperparam(json_path=r".\param.json"):
    with open(json_path, 'r') as f:
        params = json.load(f)
    with open(r".\config\schema.json", 'r') as f:
        schema = json.load(f)
    validate(instance=params, schema=schema)
    return params


def train_tokenizer():
    hyperparams = load_hyperparam(r".\config\config.json")
    dataloader = Dataloader(hyperparams["data_path"], hyperparams["allowed_chars"])
    tokenizer = Tokenizer()
    data_df = dataloader.get_df()
    tokenizer.train(data_df, hyperparams["vocab_size"])
    tokenizer.save()


if __name__ == '__main__':
    train_tokenizer()
