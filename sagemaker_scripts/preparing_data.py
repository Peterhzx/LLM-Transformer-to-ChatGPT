import gc

from tqdm import tqdm

from src.data_container import DataContainer
from tokenizer import BPE, WordPiece


def preparing_data():
    allowed_chars = [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,124,160,167,169,8208,8209,8211,8212,8216,8217,8220,8221,8224,8225,8230,8240,8242,8243,8364,171,178,179,187,192,194,198,199,200,201,202,203,206,207,212,217,219,220,224,226,230,231,232,233,234,235,238,239,244,249,251,252,255,338,339,376,691,738,7496,7497,8239,8722]
    data_container = DataContainer("../data/en-fr_5M.parquet", allowed_chars=allowed_chars)
    tok = BPE()
    tok.load("../trained_tokenizer/transformer")
    src_tokenizer_regex = "\\d+(?:[\\.,]\\d+)* | \\w+(?:[-']\\w+)* | \\S\\S+ | \\S"
    tgt_tokenizer_regex = "\\d+(?:[\\.,]\\d+)* | [a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]['’] | [a-zA-ZÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸ]+(?:[-'’][a-zA-ZÀÂÆÇÈÉÊËÎÏÔÙÛÜàâæçèéêëîïôùûüÿŒœŸ]+)* | \\S\\S+ | \\S"
    tok.tokenize(data_container.df, src_tokenizer_regex, tgt_tokenizer_regex)
    prune_and_padding_for_transformer(data_container.df, 200)
    gc.collect()
    data_container.save("../data/en-fr_5M_len200.parquet")


def prune_and_padding_for_transformer(df, max_len, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    column_list = df.columns.tolist()
    df.rename(columns={column_list[0]: "src_input", column_list[1]: "decoder_input"}, inplace=True)
    df["target"] = df["decoder_input"]
    for idx in tqdm(range(len(df)), desc="Postprocessing tokenized data", total=len(df)):
        src_tokenized = df.at[idx, "src_input"]
        tgt_tokenized = df.at[idx, "decoder_input"]

        # prune and padding
        src_input = src_tokenized[:max_len]
        src_padding = [pad_token_id] * (max_len - len(src_input))
        src_input = src_input + src_padding

        tgt_input = tgt_tokenized[:max_len - 1]

        decoder_input = [bos_token_id] + tgt_input
        decoder_input += [pad_token_id] * (max_len - len(decoder_input))

        target = tgt_input + [eos_token_id]
        target += [pad_token_id] * (max_len - len(target))

        df.at[idx, "src_input"] = src_input
        df.at[idx, "decoder_input"] = decoder_input
        df.at[idx, "target"] = target


if __name__ == '__main__':
    preparing_data()
    # data_container = DataContainer("../data/en-fr.csv", allowed_chars="")
    # df = data_container.get_df(n=5000000)
    # df.to_parquet("../data/en-fr_5M.parquet", engine="pyarrow")
