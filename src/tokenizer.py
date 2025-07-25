import base64
import hashlib
import json
import os
import time

import pandas as pd
import re
from tqdm import tqdm


# TODO: add tokenize with token_ranks mode

class Tokenizer:
    def __init__(self):
        self.tokens = {}  # {"word": int}
        self.reversed_tokens = {}  # {int: "word"}
        self.words_count = {}  # {"word": int}
        self.tokenized_words = {}  # {"word": ["w", "or", "d"]}
        self.byte_pair_count = {}  # {"or": int}
        self.byte_pair_location = {}  # {"an": {"anchor": [0], "banana": [1, 3]}}
        # self.token_ranks = {}  # {"word": int(priority)}

    @staticmethod
    def _word_level_tokenizer(text, pattern):
        tokens = pattern.findall(text)
        return tokens

    def _df_splitting(self, df, training, src_tokenizer_regex, tgt_tokenizer_regex):
        if training:
            src_tokenizer_regex = src_tokenizer_regex.split("|")
            src_tokenizer_regex = "|".join(src_tokenizer_regex[:-2])

            tgt_tokenizer_regex = tgt_tokenizer_regex.split("|")
            tgt_tokenizer_regex = "|".join(tgt_tokenizer_regex[:-2])

        # avoid modifying the original dataset
        df_word_level_tokenized = pd.DataFrame()
        col_names = df.columns.tolist()
        src_pattern = re.compile(src_tokenizer_regex, re.VERBOSE)
        tgt_pattern = re.compile(tgt_tokenizer_regex, re.VERBOSE)
        tqdm.pandas(desc=f"Splitting col {col_names[0]} into words")
        df_word_level_tokenized[col_names[0]] = df[col_names[0]].progress_apply(self._word_level_tokenizer, pattern=src_pattern)
        tqdm.pandas(desc=f"Splitting col {col_names[1]} into words")
        df_word_level_tokenized[col_names[1]] = df[col_names[1]].progress_apply(self._word_level_tokenizer, pattern=tgt_pattern)
        return df_word_level_tokenized

    def _preprocess(self, df, src_tokenizer_regex, tgt_tokenizer_regex, all_chars, special_token_list):
        if isinstance(all_chars, list):
            all_chars = sorted(set(all_chars))
            all_chars = ''.join(chr(cp) for cp in all_chars)

        # initialize tokens and reversed_tokens dict
        for i in range(len(special_token_list)):
            self.tokens[special_token_list[i]] = i
            self.reversed_tokens[i] = special_token_list[i]

        index = len(self.tokens)
        self.tokens["<EOW>"] = index
        self.reversed_tokens[index] = "<EOW>"

        index = len(self.tokens)
        for char in all_chars:
            self.tokens[char] = index
            self.reversed_tokens[index] = char
            index = index + 1

        # split data into list of words. do not include symbols.
        df_word_level_tokenized = self._df_splitting(df, True, src_tokenizer_regex, tgt_tokenizer_regex)

        # initialize words_count and words_split
        col_names = df_word_level_tokenized.columns.tolist()
        for col_name in col_names:
            for sentence in tqdm(df_word_level_tokenized[col_name], desc=f"Counting words in col {col_name}", total=len(df_word_level_tokenized)):
                for word in sentence:
                    if word not in self.words_count:
                        self.words_count[word] = 1
                        char_list = list(word)
                        char_list.append("<EOW>")
                        self.tokenized_words[word] = char_list
                    else:
                        self.words_count[word] += 1

        # initialize byte_pair_count and byte_pair_location
        for k, v in tqdm(self.tokenized_words.items(), desc="Counting and locating byte-pairs", total=len(self.tokenized_words)):
            for i in range(len(v) - 1):
                byte_pair = v[i] + v[i + 1]
                if byte_pair not in self.byte_pair_count:
                    self.byte_pair_count[byte_pair] = self.words_count[k]
                    self.byte_pair_location[byte_pair] = {k: [i]}
                else:
                    self.byte_pair_count[byte_pair] += self.words_count[k]
                    if k not in self.byte_pair_location[byte_pair]:
                        self.byte_pair_location[byte_pair][k] = [i]
                    else:
                        self.byte_pair_location[byte_pair][k].append(i)

    def _merge_byte_pair(self):
        new_token = max(self.byte_pair_count, key=self.byte_pair_count.get)
        index = len(self.tokens)
        self.tokens[new_token] = index
        self.reversed_tokens[index] = new_token

        # update byte_pair_location and byte_pair_count
        for k, v in list(self.byte_pair_location[new_token].items()):

            word_list = self.tokenized_words[k]
            self.tokenized_words[k] = self._tokenize_word(word_list, True)

            # delete previous count and location
            for i in range(len(word_list) - 1):
                byte_pair = word_list[i] + word_list[i + 1]
                if k in self.byte_pair_location[byte_pair]:
                    self.byte_pair_count[byte_pair] -= self.words_count[k] * len(self.byte_pair_location[byte_pair][k])
                    if self.byte_pair_count[byte_pair] == 0:
                        del self.byte_pair_count[byte_pair]
                    del self.byte_pair_location[byte_pair][k]

            # add new count and location
            for i in range(len(self.tokenized_words[k]) - 1):
                byte_pair = self.tokenized_words[k][i] + self.tokenized_words[k][i + 1]
                if byte_pair not in self.byte_pair_count:
                    self.byte_pair_count[byte_pair] = self.words_count[k]
                    self.byte_pair_location[byte_pair] = {k: [i]}
                else:
                    self.byte_pair_count[byte_pair] += self.words_count[k]
                    if k not in self.byte_pair_location[byte_pair]:
                        self.byte_pair_location[byte_pair][k] = [i]
                    else:
                        self.byte_pair_location[byte_pair][k].append(i)
        del self.byte_pair_location[new_token]

    def _train_loop(self, vocab_size):
        for _ in tqdm(range(vocab_size - len(self.tokens)), desc="Training", total=vocab_size - len(self.tokens)):
            self._merge_byte_pair()

    def _tokenize_word(self, word, training):
        tokenized_word = []
        matched_byte_pair = ""
        if training:
            for char in word:
                if (matched_byte_pair + char) in self.tokens:
                    matched_byte_pair += char
                else:
                    tokenized_word.append(matched_byte_pair)
                    matched_byte_pair = char
            tokenized_word.append(matched_byte_pair)
            return tokenized_word
        else:
            for char in word:
                if (matched_byte_pair + char) in self.tokens:
                    matched_byte_pair += char
                else:
                    tokenized_word.append(self.tokens.get(matched_byte_pair, 0))
                    matched_byte_pair = char
            if (matched_byte_pair + "<EOW>") in self.tokens:
                matched_byte_pair += "<EOW>"
            else:
                tokenized_word.append(self.tokens.get(matched_byte_pair, 0))
                matched_byte_pair = "<EOW>"
            tokenized_word.append(self.tokens.get(matched_byte_pair, 0))
            return tokenized_word

    def _tokenize_df(self, df, src_tokenizer_regex, tgt_tokenizer_regex):
        tokenized_df = self._df_splitting(df, False, src_tokenizer_regex, tgt_tokenizer_regex)
        col_names = tokenized_df.columns.tolist()
        for col_name in col_names:
            tokenized_col = []
            for sentence in tqdm(tokenized_df[col_name], desc=f"tokenizing col {col_name}", total=len(tokenized_df)):
                tokenized_sent = []
                for word in sentence:
                    tokenized_word = self._tokenize_word(word, False)
                    tokenized_sent += tokenized_word
                tokenized_col.append(tokenized_sent)
            tokenized_df[col_name] = tokenized_col
        return tokenized_df

    def train(self, df, vocab_size, src_tokenizer_regex, tgt_tokenizer_regex, all_chars, special_token_list):
        print("Preprocessing...")
        self._preprocess(df, src_tokenizer_regex, tgt_tokenizer_regex, all_chars, special_token_list)
        print("Training...")
        time.sleep(1)
        self._train_loop(vocab_size)

    def tokenize(self, df, src_tokenizer_regex, tgt_tokenizer_regex):
        print(f"Tokenizing DataFrame using {self._vocab_fingerprint(self.tokens)} vocab")
        tokenized_df = self._tokenize_df(df, src_tokenizer_regex, tgt_tokenizer_regex)
        return tokenized_df

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
