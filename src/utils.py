import json
import sys

import jsonschema
from jsonschema import validate

from src.dataloader import Dataloader
from src.evaluator import Evaluator
from src.trainer import Trainer
import tokenizer as tok


class NLPModelPipeline:
    def __init__(self, config_path=r".\config\config_trans.json"):
        self.params = None
        self.dataloader = None
        self.tokenizer = None
        self.trainer = None
        self.evaluator = None
        self.test_loader = None
        self._load_config(config_path)

    def _load_config(self, json_path):
        print("Loading configuration...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        with open(r".\config\schema.json", 'r') as f:
            schema = json.load(f)
        try:
            validate(instance=self.params, schema=schema)
            print("Valid configuration!")
        except jsonschema.exceptions.ValidationError as err:
            print(f"Invalid configuration: {err}")
            sys.exit(1)

    def _load_data(self):
        print("Creating Dataloader...")
        self.dataloader = Dataloader(self.params["Dataloader"])

    def _prepare_tokenizer(self):
        print("Preparing tokenizer...")
        self.tokenizer = getattr(tok, self.params["Tokenizer"]["type"])()
        if self.params["Tokenizer"]["load"]["value"]:
            self.tokenizer.load(self.params["Tokenizer"]["load"]["dir"])
        else:
            data_df = self.dataloader.get_df(self.params["Tokenizer"]["sample_size"])
            self.tokenizer.train(data_df, self.params["Tokenizer"])
            self.tokenizer.save(self.params["Tokenizer"]["load"]["dir"])

    def _tokenize_data(self):
        print("Tokenizing data...")
        self.dataloader.tokenize_df(self.tokenizer, self.params["Tokenizer"])

    def _train_and_save_model(self):
        print("Training model...")
        loader = getattr(self.dataloader, self.params["Trainer"]["dataloader"]["type"])
        train_loader, val_loader, self.test_loader = loader(**self.params["Trainer"]["dataloader"]["params"])
        self.trainer = Trainer(self.params["Trainer"])
        self.trainer.train(train_loader, val_loader)
        self.trainer.save()

    def _eval_model(self):
        print("Evaluating model...")
        self.evaluator = Evaluator(self.params["Evaluator"], self.tokenizer, self.trainer)
        self.evaluator.evaluate(self.test_loader)

    def run_pipline(self):
        self._load_data()
        self._prepare_tokenizer()
        self._tokenize_data()
        self._train_and_save_model()
        self._eval_model()
