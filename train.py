import json

from jsonschema import validate


def load_hyperparam(json_path=r".\param.json"):
    with open(json_path, 'r') as f:
        params = json.load(f)
    with open(r".\config\schema.json", 'r') as f:
        schema = json.load(f)
    validate(instance=params, schema=schema)
    return params


if __name__ == '__main__':
    hyperparams = load_hyperparam(r".\config\config.json")
