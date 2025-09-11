import gc
import json
import os.path
import sys

import jsonschema
from jsonschema import validate

from src.dataloader import Dataloader
from src.evaluator import Evaluator
import trainer as tr
import tokenizer as tok


class NLPModelPipeline:
    def __init__(self, config_path=r"./config/configs/config_trans.json", mode="local"):
        self.params = None
        self.dataloader = None
        self.tokenizer = None
        self.trainer = None
        self.evaluator = None
        self.test_loader = None
        self.mode = mode
        self._load_config(config_path)

    def _load_config(self, json_path):
        print("Loading configuration...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        with open(r"./config/schema/schema.json", 'r') as f:
            schema = json.load(f)
        try:
            validate(instance=self.params, schema=schema)
            print("Valid configuration!")
        except jsonschema.exceptions.ValidationError as err:
            print(f"Invalid configuration: {err}")
            sys.exit(1)

    def _load_data(self):
        print("Creating Dataloader...")
        self.dataloader = Dataloader(**self.params["Dataloader"], mode=self.mode)

    def _prepare_tokenizer(self):
        print("Preparing tokenizer...")
        tokenizer_type = self.params["Tokenizer"]["type"]
        self.tokenizer = getattr(tok, tokenizer_type)()

        load_params = self.params["Tokenizer"].get("load", {})
        load_value = load_params.get("value", False)
        tokenizer_dir = load_params.get("dir", os.path.join("./trained_tokenizer", tokenizer_type.lower()))

        if load_value:
            # Load existing tokenizer
            self.tokenizer.load(tokenizer_dir)
        else:
            # Train new tokenizer
            sample_size = self.params["Tokenizer"].get("sample_size", 0.1)
            data_df = self.dataloader.get_df(sample_size)
            self.tokenizer.train(data_df, **self.params["Tokenizer"])
            del data_df
            gc.collect()

            # Ensure directory exists
            os.makedirs(tokenizer_dir, exist_ok=True)
            self.tokenizer.save(tokenizer_dir)

    def _tokenize_data(self):
        print("Tokenizing data...")
        self.tokenizer.tokenize(self.dataloader.df, **self.params["Tokenizer"])

    def _train_and_save_model(self):
        print("Training model...")
        train_loader, val_loader, self.test_loader = self.dataloader.get_dataloader(**self.params["Trainer"]["dataloader"])
        _trainer = getattr(tr, self.params["Trainer"]["type"])
        if self.tokenizer:
            num_tokens = len(self.tokenizer)
        else:
            num_tokens = self.params["Trainer"]["num_tokens"]
        self.trainer = _trainer(**self.params["Trainer"], num_tokens=num_tokens, mode=self.mode)
        self.trainer.train(train_loader, val_loader)
        self.trainer.save()

    def _eval_model(self):
        print("Evaluating model...")
        self.evaluator = Evaluator(self.params["Evaluator"], self.tokenizer, self.trainer)
        self.evaluator.evaluate(self.test_loader)

    def run_pipline(self):
        self._load_data()
        if "Tokenizer" in self.params:
            self._prepare_tokenizer()
            self._tokenize_data()
            self._train_and_save_model()
            # self._eval_model()
        else:
            print("Training without tokenizer")
            self._train_and_save_model()
            # self._eval_model()

        """
        self._load_data()
        self._prepare_tokenizer()
        self._tokenize_data()
        self.dataloader.save("./data/en-fr_tokenized.scv")
        
        
        self._load_data()
        if "Tokenizer" in self.params:
            self._prepare_tokenizer()
            self._tokenize_data()
            self._train_and_save_model()
            self._eval_model()
        else:
            print("Training without tokenizer")
            self._train_and_save_model()
            self._eval_model()
        """
