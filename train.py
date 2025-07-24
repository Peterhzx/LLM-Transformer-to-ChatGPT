import json

from jsonschema import validate


def load_hyperparam(json_path="./param.json"):
    with open(json_path, 'r') as f:
        params = json.load(f)
    with open("./config/schema.json", 'r') as f:
        schema = json.load(f)
    validate(instance=params, schema=schema)
    return params


if __name__ == '__main__':
    hyperparams = load_hyperparam("./config/config.json")
