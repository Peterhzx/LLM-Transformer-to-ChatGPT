import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self, df, max_len, pad_token_id=0, bos_token_id=1, eos_token_id=2):
        super(TransformerDataset, self).__init__()
        self.df = df
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # improvement: Instead of fixed-length sequences, consider dynamic batching with a custom collate_fn
        row = self.df.iloc[idx]
        col_names = self.df.columns.tolist()
        src_tokenized = row[col_names[0]]
        tgt_tokenized = row[col_names[1]]

        # prune and padding
        src_input = src_tokenized[:self.max_len]
        src_padding = [self.pad_token_id] * (self.max_len - len(src_input))
        src_input = src_input + src_padding

        tgt_input = tgt_tokenized[:self.max_len - 1]

        decoder_input = [self.bos_token_id] + tgt_input
        decoder_input += [self.pad_token_id] * (self.max_len - len(decoder_input))

        target = tgt_input + [self.eos_token_id]
        target += [self.pad_token_id] * (self.max_len - len(target))

        src_tensor = torch.tensor(src_input, dtype=torch.long)
        decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return src_tensor, decoder_input_tensor, target_tensor


class BERTDataset(Dataset):
    def __init__(self, df, max_len, pad_token_id=0, mask_token_id=1, cls_token_id=2, sep_token_id=3, isnext_token_id=4, notnext_token_id=5):
        super(BERTDataset, self).__init__()
        self.df = df
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.isnext_token_id = isnext_token_id
        self.notnext_token_id = notnext_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # improvement: Instead of fixed-length sequences, consider dynamic batching with a custom collate_fn
        row = self.df.iloc[idx]
        col_names = self.df.columns.tolist()
        src_tokenized = row[col_names[0]]
        tgt_tokenized = row[col_names[1]]

        # prune and padding
        src_input = src_tokenized[:self.max_len]
        src_padding = [self.pad_token_id] * (self.max_len - len(src_input))
        src_input = src_input + src_padding

        tgt_input = tgt_tokenized[:self.max_len - 1]

        decoder_input = [self.bos_token_id] + tgt_input
        decoder_input += [self.pad_token_id] * (self.max_len - len(decoder_input))

        target = tgt_input + [self.eos_token_id]
        target += [self.pad_token_id] * (self.max_len - len(target))

        src_tensor = torch.tensor(src_input, dtype=torch.long)
        decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return src_tensor, decoder_input_tensor, target_tensor