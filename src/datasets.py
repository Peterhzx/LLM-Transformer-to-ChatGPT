import ast
import random

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

        if isinstance(src_tokenized, str) or isinstance(tgt_tokenized, str):
            src_tokenized = ast.literal_eval(src_tokenized)
            tgt_tokenized = ast.literal_eval(tgt_tokenized)

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
    def __init__(self, df, max_len, random_token_start, random_token_end, pad_token_id=0, mask_token_id=1, cls_token_id=2, sep_token_id=3):
        super(BERTDataset, self).__init__()
        self.df = df
        self.max_len = max_len
        self.random_token_start = random_token_start
        self.random_token_end = random_token_end
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # improvement: Instead of fixed-length sequences, consider dynamic batching with a custom collate_fn
        a_tokenized = self.df.iloc[idx, 0]
        if random.random() > 0.5:
            b_tokenized = self.df.iloc[idx, 1]
            isnext_token = 1
        else:
            i = random.randint(0, len(self.df)-2)
            b_tokenized = self.df.iloc[i + 1 if i >= idx else i, 1]
            isnext_token = 0

        # create tgt seq
        tgt_tokenized = [isnext_token] + a_tokenized + [self.sep_token_id] + b_tokenized
        target = tgt_tokenized[:self.max_len]
        target += [self.pad_token_id] * (self.max_len - len(target))

        # mask src
        length = len(a_tokenized) + len(b_tokenized)
        if length > self.max_len - 2:
            length = self.max_len - 2
        pred_list = random.sample(range(length), int(length * 0.15) if int(length * 0.15) > 0 else 1)

        for i in range(len(pred_list)):
            if pred_list[i] >= len(a_tokenized):
                pred_list[i] += 2
            else:
                pred_list[i] += 1

        src_tokenized = [self.cls_token_id] + a_tokenized + [self.sep_token_id] + b_tokenized

        mask_list = random.sample(pred_list, int(len(pred_list) * 0.9) if int(len(pred_list) * 0.9) > 0 else 1)
        for i in mask_list:
            src_tokenized[i] = self.mask_token_id

        random_list = random.sample(mask_list, int(len(mask_list) * (1 / 9)) if int(len(mask_list) * (1 / 9)) > 0 else 1)
        for i in random_list:
            src_tokenized[i] = random.randint(self.random_token_start, self.random_token_end)

        source = src_tokenized[:self.max_len]
        source += [self.pad_token_id] * (self.max_len - len(source))

        mask = [False] * self.max_len
        for i in pred_list:
            mask[i] = True

        # to tensor
        src_tensor = torch.tensor(source, dtype=torch.long)
        tgt_tensor = torch.tensor(target, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        return src_tensor, tgt_tensor, mask_tensor
