import gc
import math
import re
import time

from tqdm import tqdm

from tokenizer import Tokenizer


class WordPiece(Tokenizer):
    def __init__(self):
        super(WordPiece, self).__init__()
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
        special_tokens = config.get("special_tokens", [])
        print("Preprocessing data for training tokenizer")
        time.sleep(0.5)
        self._preprocess(df, src_tokenizer_regex, tgt_tokenizer_regex, all_chars)
        gc.collect()
        print("Training tokenizer")
        time.sleep(0.5)
        self._train_loop(vocab_size - len(special_tokens))
        print("Postprocessing")
        self._postprocess(special_tokens)

    def tokenize(self, df, config, **kwargs):
        src_tokenizer_regex = config["src_tokenizer_regex"]
        tgt_tokenizer_regex = config["tgt_tokenizer_regex"]
        print(f"Tokenizing DataFrame using {self._vocab_fingerprint(self.tokens)} vocab")
        time.sleep(0.5)
        self._df_splitting(df, False, src_tokenizer_regex, tgt_tokenizer_regex)
        gc.collect()
        self._tokenize_df(df)
        gc.collect()

    def _preprocess(self, df, src_tokenizer_regex, tgt_tokenizer_regex, all_chars):
        if isinstance(all_chars, list):
            all_chars = sorted(set(all_chars))
            all_chars = ''.join(chr(cp) for cp in all_chars)

        # initialize tokens
        index = 0
        for char in all_chars:
            self.tokens[char] = index
            index = index + 1

        # split data into list of words. do not include symbols.
        self._df_splitting(df, True, src_tokenizer_regex, tgt_tokenizer_regex)

        # initialize words_count and tokenized_words
        col_names = df.columns.tolist()
        for col_name in col_names:
            for sentence in tqdm(df[col_name], desc=f"Counting words in col {col_name}", total=len(df)):
                for word in sentence:
                    if word not in self.words_count:
                        self.words_count[word] = 1
                        self.tokenized_words[word] = list(word)
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
    def _log_likelihood(counter, total):
        ll = 0.0
        for tok, c in counter.items():
            p = c / total
            ll += c * math.log(p)
        return ll

    def _delta_ll(self, new_token_count, old_left_part_count, old_right_part_count):
        def term(c, total):
            return 0.0 if c <= 0 else c * math.log(c / total)

        old_total = self.total_count
        new_total = old_total - new_token_count

        # new contributions
        ll = term(new_token_count, new_total)
        ll += term(old_left_part_count - new_token_count, new_total)
        ll += term(old_right_part_count - new_token_count, new_total)

        # subtract old contributions
        ll -= term(old_left_part_count, old_total)
        ll -= term(old_right_part_count, old_total)

        return ll

    def _df_splitting(self, df, training, src_tokenizer_regex, tgt_tokenizer_regex):
        if training:
            src_tokenizer_regex = src_tokenizer_regex.split("|")
            src_tokenizer_regex = "|".join(src_tokenizer_regex[:-2])

            tgt_tokenizer_regex = tgt_tokenizer_regex.split("|")
            tgt_tokenizer_regex = "|".join(tgt_tokenizer_regex[:-2])

        col_names = df.columns.tolist()
        src_pattern = re.compile(src_tokenizer_regex, re.VERBOSE)
        tgt_pattern = re.compile(tgt_tokenizer_regex, re.VERBOSE)

        tqdm.pandas(desc=f"Splitting col {col_names[0]} into words")
        tokenized_src = df[col_names[0]].progress_apply(self._word_level_tokenizer, pattern=src_pattern)
        df[col_names[0]] = tokenized_src
        del tokenized_src
        gc.collect()
        tqdm.pandas(desc=f"Splitting col {col_names[1]} into words")
        tokenized_tgt = df[col_names[1]].progress_apply(self._word_level_tokenizer, pattern=tgt_pattern)
        df[col_names[1]] = tokenized_tgt
        del tokenized_tgt
        gc.collect()

    def _train_loop(self, vocab_size):
        with tqdm(total=vocab_size - len(self.tokens), desc="Training") as pbar:
            while len(self.tokens) < vocab_size and self.byte_pair_count:
                self._merge_byte_pair()
                pbar.update(1)
                pbar.set_postfix(log_likelihood=self._log_likelihood(self.tokens_count, self.total_count))
                gc.collect()

    def _merge_byte_pair(self):
        # compute delta likelihood
        ll_gain = {}

        try:
            for k, v in self.byte_pair_count.items():
                word, loc_list = next(iter(self.byte_pair_location[k].items()))
                location = loc_list[0]
                left_part = self.tokenized_words[word][location]
                right_part = self.tokenized_words[word][location+1]

                old_left_part_count = self.tokens_count[left_part]
                old_right_part_count = self.tokens_count[right_part]

                ll_gain[k] = self._delta_ll(v, old_left_part_count, old_right_part_count)
        except KeyError as e:
            print(f"KeyError {e} in compute delta likelihood")

        new_token = max(ll_gain, key=ll_gain.get)

        index = len(self.tokens)
        self.tokens[new_token] = index

        # update tokens_count and total_count
        try:
            word, loc_list = next(iter(self.byte_pair_location[new_token].items()))
            location = loc_list[0]
            left_part = self.tokenized_words[word][location]
            right_part = self.tokenized_words[word][location + 1]
            self.tokens_count[new_token] = self.byte_pair_count[new_token]
            for part in (left_part, right_part):
                if part in self.tokens_count:
                    self.tokens_count[part] -= self.byte_pair_count[new_token]
                    if self.tokens_count[part] <= 0:
                        del self.tokens_count[part]
            self.total_count -= self.byte_pair_count[new_token]
        except KeyError as e:
            print(f"KeyError {e} in update tokens_count and total_count")

        # update tokenized_words, byte_pair_location and byte_pair_count
        for k, v in list(self.byte_pair_location[new_token].items()):

            try:
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
            except KeyError as e:
                print(f"KeyError {e} in update tokenized_words")

            # add new count and location
            try:
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
            except KeyError as e:
                print(f"KeyError {e} in add new count and location")
        del self.byte_pair_location[new_token]

    def _postprocess(self, special_tokens):
        final_vocab = {}
        appears_at_start = {}
        appears_in_the_mid = {}

        for k, v in self.tokens.items():
            appears_at_start[k] = False
            appears_in_the_mid[k] = False

        for i in range(len(special_tokens)):
            final_vocab[special_tokens[i]] = i
            self.reversed_tokens[i] = special_tokens[i]

        for word, word_list in self.tokenized_words.items():
            appears_at_start[word_list[0]] = True
            if len(word_list) > 1:
                for i in range(len(word_list)-1):
                    appears_in_the_mid[word_list[i+1]] = True

        for k, v in appears_at_start.items():
            if v:
                index = len(final_vocab)
                final_vocab[k] = index
                self.reversed_tokens[index] = k

        for k, v in appears_in_the_mid.items():
            if v:
                index = len(final_vocab)
                final_vocab["##" + k] = index
                self.reversed_tokens[index] = "##" + k

        self.tokens = final_vocab
        print(f"Final Vocabulary has {len(self.tokens)} tokens")

    def _tokenize_df(self, df):
        col_names = df.columns.tolist()
        for col_name in col_names:
            tokenized_col = []
            for sentence in tqdm(df[col_name], desc=f"tokenizing col {col_name}", total=len(df)):
                if not sentence:
                    sentence = [" "]
                tokenized_sent = []
                for word in sentence:
                    tokenized_word = self._tokenize_word(word, False, self.tokens.get("<UNK>", 4))
                    tokenized_sent += tokenized_word
                tokenized_col.append(tokenized_sent)
            df[col_name] = tokenized_col

    def _tokenize_word(self, word, training, unk_token_id=4):
        tokenized_word = []
        if training:
            matched_byte_pair = ""
            for char in word:
                if (matched_byte_pair + char) in self.tokens:
                    matched_byte_pair += char
                else:
                    tokenized_word.append(matched_byte_pair)
                    matched_byte_pair = char
            tokenized_word.append(matched_byte_pair)
            return tokenized_word
        else:
            start = 0
            end = len(word)
            while start < end:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                if sub in self.tokens:
                    tokenized_word.append(self.tokens.get(sub, unk_token_id))
                    start = end
                    end = len(word)
                else:
                    end -= 1
                    if start == end:
                        return [unk_token_id]
            return tokenized_word
