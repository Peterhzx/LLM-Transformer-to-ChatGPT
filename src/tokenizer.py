import pandas as pd
import re
from tqdm import tqdm


class Tokenizer:
    def __init__(self):
        self.words_dic = {}
        self.tokens = {}
        self.words_count = {}
        self.words_split = {}

    @staticmethod
    def word_level_tokenizer(text, regex):
        pattern = re.compile(regex, re.VERBOSE)
        tokens = pattern.findall(text)
        return tokens

    def preprocess(self, df):
        en_tokenizer_regex = r"""
              \d+(?:[\.,]\d+)*
            | \w+(?:[-']\w+)*
            """

        fr_tokenizer_regex = r"""
              \d+(?:[\.,]\d+)*
            | [a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]['’]
            | [a-zA-ZÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸ]+(?:[-'’][a-zA-ZÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸ]+)*
            """

        df_word_level_tokenized = pd.DataFrame()
        tqdm.pandas()
        df_word_level_tokenized["en_tokens"] = df["en"].progress_apply(self.word_level_tokenizer,
                                                                       regex=en_tokenizer_regex)
        df_word_level_tokenized["fr_tokens"] = df["fr"].progress_apply(self.word_level_tokenizer,
                                                                       regex=fr_tokenizer_regex)
        return df_word_level_tokenized

