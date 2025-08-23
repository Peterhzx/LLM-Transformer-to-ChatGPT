import math
import re
import sys
import time

import pandas as pd
from tqdm import tqdm

from tokenizer import Tokenizer


class WordPiece(Tokenizer):
    def __init__(self):
        super(WordPiece).__init__()
        self.tokens = {}  # {"word": int}
        self.reversed_tokens = {}  # {int: "word"}
        self.tokens_count = {}  # {"word": int}
        self.total_count = 0
        self.words_count = {}  # {"word": int}
        self.tokenized_words = {}  # {"word": ["w", "or", "d"]}
        self.byte_pair_count = {}  # {"or": int}
        self.byte_pair_location = {}  # {"an": {"anchor": [0], "banana": [1, 3]}}
        # self.token_ranks = {}  # {"word": int(priority)}

    def train(self, df, config, *args, **kwargs):
        vocab_size = config["vocab_size"]
        src_tokenizer_regex = config["src_tokenizer_regex"]
        tgt_tokenizer_regex = config["tgt_tokenizer_regex"]
        all_chars = config["all_chars"]
        special_tokens = config["special_tokens"]
        print("Preprocessing data for training tokenizer")
        time.sleep(0.5)
        self._preprocess(df, src_tokenizer_regex, tgt_tokenizer_regex, special_tokens, all_chars)
        print("Training tokenizer")
        time.sleep(0.5)
        self._train_loop(vocab_size)

    def tokenize(self, df, config, **kwargs):
        src_tokenizer_regex = config["src_tokenizer_regex"]
        tgt_tokenizer_regex = config["tgt_tokenizer_regex"]
        print(f"Tokenizing DataFrame using {self._vocab_fingerprint(self.tokens)} vocab")
        time.sleep(0.5)
        tokenized_df = self._tokenize_df(df, src_tokenizer_regex, tgt_tokenizer_regex)
        return tokenized_df

    def _preprocess(self, df, src_tokenizer_regex, tgt_tokenizer_regex, special_token_list, all_chars):
        if isinstance(all_chars, list):
            all_chars = sorted(set(all_chars))
            all_chars = ''.join(chr(cp) for cp in all_chars)

        # initialize tokens and reversed_tokens dict
        for i in range(len(special_token_list)):
            self.tokens[special_token_list[i]] = i
            self.reversed_tokens[i] = special_token_list[i]

        index = len(self.tokens)
        self.tokens["<SOW>"] = index
        self.reversed_tokens[index] = "<SOW>"

        index = len(self.tokens)
        for char in all_chars:
            self.tokens[char] = index
            self.reversed_tokens[index] = char
            index = index + 1

        # split data into list of words. do not include symbols.
        df_word_level_tokenized = self._df_splitting(df, True, src_tokenizer_regex, tgt_tokenizer_regex)

        # initialize words_count and tokenized_words
        col_names = df_word_level_tokenized.columns.tolist()
        for col_name in col_names:
            for sentence in tqdm(df_word_level_tokenized[col_name], desc=f"Counting words in col {col_name}", total=len(df_word_level_tokenized)):
                for word in sentence:
                    if word not in self.words_count:
                        self.words_count[word] = 1
                        char_list = ["<SOW>"]
                        char_list += list(word)
                        self.tokenized_words[word] = char_list
                    else:
                        self.words_count[word] += 1

        # initialize tokens_count, byte_pair_count and byte_pair_location
        for k, v in tqdm(self.tokenized_words.items(), desc="Counting and locating byte-pairs", total=len(self.tokenized_words)):
            for i in range(len(v) - 1):
                if v[i] not in self.tokens_count:
                    self.tokens_count[v[i]] = self.words_count[k]
                else:
                    self.tokens_count[v[i]] += self.words_count[k]
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
            if v[-1] not in self.tokens_count:
                self.tokens_count[v[-1]] = self.words_count[k]
            else:
                self.tokens_count[v[-1]] += self.words_count[k]

        # initialize total_count
        self.total_count = sum(self.tokens_count.values())

    @staticmethod
    def _word_level_tokenizer(text, pattern):
        tokens = pattern.findall(text)
        return tokens

    @staticmethod
    def log_likelihood(counter, total):
        ll = 0.0
        for tok, c in counter.items():
            p = c / total
            ll += c * math.log(p)
        return ll

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
        df_word_level_tokenized[col_names[0]] = df[col_names[0]].progress_apply(self._word_level_tokenizer,
                                                                                pattern=src_pattern)
        tqdm.pandas(desc=f"Splitting col {col_names[1]} into words")
        df_word_level_tokenized[col_names[1]] = df[col_names[1]].progress_apply(self._word_level_tokenizer,
                                                                                pattern=tgt_pattern)
        return df_word_level_tokenized

    def _train_loop(self, vocab_size):
        total_ll = self.log_likelihood(self.tokens_count, self.total_count)
        with tqdm(total=vocab_size - len(self.tokens), desc="Training") as pbar:
            while len(self.tokens) < vocab_size:
                total_ll = self._merge_byte_pair(vocab_size, pbar, total_ll)

    def _merge_byte_pair(self, vocab_size, pbar, total_ll):
        # compute likelihood
        ll_gain = {}

        for k, v in self.byte_pair_count.items():
            new_tokens_count = self.tokens_count.copy()
            new_tokens_count[k] = v

            word, loc_list = next(iter(self.byte_pair_location[k].items()))
            location = loc_list[0]
            left_part = self.tokenized_words[word][location]
            right_part = self.tokenized_words[word][location+1]

            new_tokens_count[left_part] -= v
            new_tokens_count[right_part] -= v
            if new_tokens_count[left_part] <= 0:
                del new_tokens_count[left_part]
            if new_tokens_count[right_part] <= 0:
                del new_tokens_count[right_part]

            ll_gain[k] = self.log_likelihood(new_tokens_count, self.total_count - v) - total_ll

        new_token = max(ll_gain, key=ll_gain.get)
        total_ll += ll_gain[new_token]

        index = len(self.tokens)
        self.tokens[new_token] = index
        self.reversed_tokens[index] = new_token
        pbar.update(1)

        if not new_token.startswith("<SOW>") and len(self.tokens) < vocab_size:
            index += 1
            sub_word_token = "##" + new_token
            self.tokens[sub_word_token] = index
            self.reversed_tokens[index] = sub_word_token
            pbar.update(1)

        # update tokens_count and total_count
        word, loc_list = next(iter(self.byte_pair_location[new_token].items()))
        location = loc_list[0]
        left_part = self.tokenized_words[word][location]
        right_part = self.tokenized_words[word][location + 1]
        self.tokens_count[new_token] = self.byte_pair_count[new_token]
        self.tokens_count[left_part] -= self.byte_pair_count[new_token]
        self.tokens_count[right_part] -= self.byte_pair_count[new_token]
        self.total_count -= self.byte_pair_count[new_token]
        if self.tokens_count[left_part] <= 0:
            del self.tokens_count[left_part]
        if self.tokens_count[right_part] <= 0:
            del self.tokens_count[right_part]

        # update tokenized_words, byte_pair_location and byte_pair_count
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
        return total_ll

    def _tokenize_df(self, df, src_tokenizer_regex, tgt_tokenizer_regex):
        tokenized_df = self._df_splitting(df, False, src_tokenizer_regex, tgt_tokenizer_regex)
        col_names = tokenized_df.columns.tolist()
        for col_name in col_names:
            tokenized_col = []
            for sentence in tqdm(tokenized_df[col_name], desc=f"tokenizing col {col_name}", total=len(tokenized_df)):
                if len(sentence) == 0:
                    sentence.append(" ")
                tokenized_sent = []
                for word in sentence:
                    tokenized_word = self._tokenize_word(word, False)
                    tokenized_sent += tokenized_word
                tokenized_col.append(tokenized_sent)
            tokenized_df[col_name] = tokenized_col
        return tokenized_df

    def _tokenize_word(self, word, training):
        tokenized_word = []
        matched_byte_pair = "" if training else "<SOW>"
        for char in word:
            if (matched_byte_pair + char) in self.tokens:
                matched_byte_pair += char
            else:
                token = matched_byte_pair if training else self.tokens.get(matched_byte_pair, 0)
                if not training and tokenized_word:
                    token = "##" + str(token)
                tokenized_word.append(token)
                matched_byte_pair = char
        token = matched_byte_pair if training else self.tokens.get(matched_byte_pair, 0)
        if not training and tokenized_word:
            token = "##" + str(token)
        tokenized_word.append(token)
        return tokenized_word
