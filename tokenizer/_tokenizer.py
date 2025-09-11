import base64
import hashlib
import json
import os
from abc import ABC, abstractmethod


class Tokenizer(ABC):

    def __init__(self):
        super(ABC, self).__init__()
        self.reversed_tokens = None
        self.tokens = None

    @abstractmethod
    def train(self, df, vocab_size, src_tokenizer_regex, tgt_tokenizer_regex, all_chars, special_tokens, *args, **kwargs):
        pass

    @abstractmethod
    def tokenize(self, df, src_tokenizer_regex, tgt_tokenizer_regex, *args, **kwargs):
        pass

    @staticmethod
    def _vocab_fingerprint(d):
        hasher = hashlib.sha256()

        encoder = json.JSONEncoder(
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
            default=str
        )
        for chunk in encoder.iterencode(d):
            hasher.update(chunk.encode('utf-8'))

        # Get hash bytes and truncate
        full_hash = hasher.digest()
        hash_bytes = full_hash[:6]

        # Encode to URL-safe base64 and remove padding
        return base64.urlsafe_b64encode(hash_bytes).decode('ascii').rstrip('=')

    def save(self, path=None):
        if path is None:
            with open("./tokens.json", "w") as f:
                json.dump(self.tokens, f, indent=4)
            with open("./reversed_tokens.json", "w") as f:
                json.dump(self.reversed_tokens, f, indent=4)
            with open("./fingerprint.txt", "w") as f:
                f.write(f"tokens: {self._vocab_fingerprint(self.tokens)}\n"
                        f"reversed tokens: {self._vocab_fingerprint(self.reversed_tokens)}")
            print(f"{self._vocab_fingerprint(self.tokens)} vocab saved to {os.getcwd()}")
        else:
            with open(f"{path}/tokens.json", "w") as f:
                json.dump(self.tokens, f, indent=4)
            with open(f"{path}/reversed_tokens.json", "w") as f:
                json.dump(self.reversed_tokens, f, indent=4)
            with open(f"{path}/fingerprint.txt", "w") as f:
                f.write(f"tokens: {self._vocab_fingerprint(self.tokens)}\n"
                        f"reversed tokens: {self._vocab_fingerprint(self.reversed_tokens)}")
            print(f"{self._vocab_fingerprint(self.tokens)} vocab saved to {path}")

    def load(self, path=None):
        if path is None:
            with open("./tokens.json", "r") as f:
                self.tokens = json.load(f)
            with open("./reversed_tokens.json", "r") as f:
                self.reversed_tokens = json.load(f)
            print(f"{self._vocab_fingerprint(self.tokens)} vocab loaded from {os.getcwd()}")
        else:
            with open(f"{path}/tokens.json", "r") as f:
                self.tokens = json.load(f)
            with open(f"{path}/reversed_tokens.json", "r") as f:
                self.reversed_tokens = json.load(f)
            print(f"{self._vocab_fingerprint(self.tokens)} vocab loaded from {path}")

    def __len__(self):
        return len(self.tokens)
