import pandas as pd
import re
from tqdm import tqdm


class Tokenizer:
    def __init__(self):
        self.tokens = {}
        self.reversed_tokens = {}
        self.words_count = {}
        self.tokenized_words = {}
        # self.code_points = code_points  # list of integers of valid char in unicode format

    @staticmethod
    def _word_level_tokenizer(text, regex):
        pattern = re.compile(regex, re.VERBOSE)
        tokens = pattern.findall(text)
        return tokens

    def _preprocess(self, df):
        # including all the English and French char in unicode:
        # [ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz| §©«²³»ÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸʳˢᵈᵉ‐‑–—‘’“”†‡… ‰′″€−]
        code_points = [
            0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
            0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
            0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
            0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
            0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
            0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
            0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70,
            0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
            0x79, 0x7A, 0x7C, 0xA0, 0xA7, 0xA9,
            0x2010, 0x2011, 0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D,
            0x2020, 0x2021, 0x2026, 0x2030, 0x2032, 0x2033, 0x20AC,
            0x00AB, 0x00B2, 0x00B3, 0x00BB,
            0x00C0, 0x00C2, 0x00C6, 0x00C7, 0x00C8, 0x00C9,
            0x00CA, 0x00CB, 0x00CE, 0x00CF, 0x00D4, 0x00D9,
            0x00DB, 0x00DC, 0x00E0, 0x00E2, 0x00E6, 0x00E7,
            0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EE, 0x00EF,
            0x00F4, 0x00F9, 0x00FB, 0x00FC, 0x00FF,
            0x0152, 0x0153, 0x0178,
            0x02B3, 0x02E2,
            0x1D48, 0x1D49,
            0x202F,
            0x2212
        ]
        code_points = sorted(set(code_points))
        all_chars = ''.join(chr(cp) for cp in code_points)

        # initialize word2token and token2word dict
        special_token_list = ["<PAD>", "<BOS>", "<EOS>", "<EOW>"]
        for i in range(len(special_token_list)):
            self.tokens[special_token_list[i]] = i
            self.reversed_tokens[i] = special_token_list[i]

        index = len(self.tokens)
        for char in all_chars:
            self.tokens[char] = index
            self.reversed_tokens[index] = char
            index = index + 1

        # split data into list of words. do not include symbols.
        en_tokenizer_regex = r"""
              \d+(?:[\.,]\d+)*  # Numbers: 1.23, 1,000,000
            | \w+(?:[-']\w+)*   # Words: don't, state-of-the-art
            """

        fr_tokenizer_regex = r"""
              \d+(?:[\.,]\d+)*  # Numbers: 1.23, 1,000,000
            | [a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]['’]    # Elisions: l', d', j'
            | [a-zA-ZÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸ]+(?:[-'’][a-zA-ZÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸ]+)*  # Words w/ hyphens/apostrophes
            """

        # avoid modify the original dataset
        df_word_level_tokenized = pd.DataFrame()
        tqdm.pandas()
        col_names = df.columns.tolist()
        df_word_level_tokenized[col_names[0]] = df[col_names[0]].progress_apply(self._word_level_tokenizer,
                                                                                regex=en_tokenizer_regex,
                                                                                desc=f"Splitting col {col_names[0]} into words")
        df_word_level_tokenized[col_names[1]] = df[col_names[1]].progress_apply(self._word_level_tokenizer,
                                                                                regex=fr_tokenizer_regex,
                                                                                desc=f"Splitting col {col_names[1]} into words")

        # initialize words_count and words_split
        for col_name in col_names:
            for sentence in tqdm(df_word_level_tokenized[col_name], desc=f"Counting words in col {col_name}", total=len(df_word_level_tokenized)):
                for word in sentence:
                    if word not in self.words_count:
                        self.words_count[word] = 1
                        char_list = list(word).append("<EOW>")
                        self.tokenized_words[word] = char_list
                    else:
                        self.words_count[word] += 1

    def _count_byte_pair(self):
        byte_pair_count = {}
        for k, v in self.tokenized_words.items():
            for i in range(len(v) - 1):
                byte_pair = v[i] + v[i + 1]
                if byte_pair not in byte_pair_count:
                    byte_pair_count[byte_pair] = self.words_count[k]
                else:
                    byte_pair_count[byte_pair] += self.words_count[k]
        return byte_pair_count

    def _merge_byte_pair(self, byte_pair_count):
        new_token = max(byte_pair_count, key=byte_pair_count.get)
        index = len(self.tokens)
        self.tokens[new_token] = index
        self.reversed_tokens[index] = new_token
        new_tokenized_words = {}

        for k, v in self.tokenized_words.items():
            new_tokenized_words[k] = self._tokenize_word(v, True)
        self.tokenized_words = new_tokenized_words

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
                    tokenized_word.append(self.reversed_tokens.get(matched_byte_pair, 0))
                    matched_byte_pair = char
            if (matched_byte_pair + "<EOW>") in self.tokens:
                matched_byte_pair += "<EOW>"
            else:
                tokenized_word.append(self.reversed_tokens.get(matched_byte_pair, 0))
                matched_byte_pair = "<EOW>"
            tokenized_word.append(self.reversed_tokens.get(matched_byte_pair, 0))
            return tokenized_word
