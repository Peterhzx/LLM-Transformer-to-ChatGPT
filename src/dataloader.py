import ast
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets import TransformerDataset, BERTDataset


class Dataloader:
    def __init__(self, config, mode="local"):
        df_path = config["data_file"]
        if mode == "sagemaker":
            parent = os.environ.get('SM_CHANNEL_TRAIN', "/opt/ml/input/data/train")
            df_path = os.path.join(parent, df_path)
        allowed_chars = config.get("allowed_chars", "")
        nrows = config.get("nrows", None)

        print(f"Loading data: {df_path}")
        if df_path.endswith(".csv"):
            self.df = pd.read_csv(df_path, nrows=nrows)
        elif df_path.endswith(".xlsx"):
            self.df = pd.read_excel(df_path, nrows=nrows)
        elif df_path.endswith(".json"):
            self.df = pd.read_json(df_path, nrows=nrows)
        elif df_path.endswith(".parquet"):
            self.df = pd.read_parquet(df_path, engine='pyarrow')
        else:
            raise ValueError("file formate should be: .csv .xlsx .json .parquet")

        if isinstance(allowed_chars, list):
            code_points = sorted(set(allowed_chars))
            self.allowed_chars = ''.join(chr(cp) for cp in code_points)
        elif isinstance(allowed_chars, str):
            self.allowed_chars = allowed_chars
        else:
            raise ValueError("allowed_chars accept only list and str")

        if len(allowed_chars) != 0:
            self._clean_data()

    def _clean_data(self):
        self.df = self.df.drop_duplicates().dropna().astype(str).reset_index(drop=True)
        invalid_sentence_index = set()
        column_list = self.df.columns.tolist()

        for col_name in column_list:
            clean_col = []
            for index, sentence in tqdm(enumerate(self.df[col_name]), desc=f"Cleaning column {col_name}",
                                        total=len(self.df)):
                filtered_chars = [char for char in sentence if char in self.allowed_chars]
                clean_sentence = ''.join(filtered_chars)
                if len(clean_sentence) == 0:  # avoid nan in training
                    clean_sentence = " "
                    invalid_sentence_index.add(index)
                clean_col.append(clean_sentence)
            self.df[col_name] = clean_col

        valid_indices = [i for i in self.df.index if i not in invalid_sentence_index]
        self.df = self.df.loc[valid_indices].reset_index(drop=True)

    def save(self, path):
        self.df.to_csv(path, index=False)

    def get_df(self, sample_size=0.1):
        return self.df.sample(frac=sample_size)

    def _postprocess_tokenized_data_for_transformer(self, max_len, pad_token_id, bos_token_id, eos_token_id):
        column_list = self.df.columns.tolist()
        self.df.rename(columns={column_list[0]: "src_input", column_list[1]: "decoder_input"}, inplace=True)
        self.df["target"] = self.df["decoder_input"]
        for idx in tqdm(range(len(self.df)), desc="Postprocessing tokenized data", total=len(self.df)):
            src_tokenized = self.df.at[idx, "src_input"]
            tgt_tokenized = self.df.at[idx, "decoder_input"]

            # prune and padding
            src_input = src_tokenized[:max_len]
            src_padding = [pad_token_id] * (max_len - len(src_input))
            src_input = src_input + src_padding

            tgt_input = tgt_tokenized[:max_len - 1]

            decoder_input = [bos_token_id] + tgt_input
            decoder_input += [pad_token_id] * (max_len - len(decoder_input))

            target = tgt_input + [eos_token_id]
            target += [pad_token_id] * (max_len - len(target))

            self.df.at[idx, "src_input"] = src_input
            self.df.at[idx, "decoder_input"] = decoder_input
            self.df.at[idx, "target"] = target

    def get_transformer_dataloader(self, max_seq_len, batch_size, num_workers, pin_memory, pad_token_id=0, bos_token_id=1, eos_token_id=2, train_val_test_split=None):
        self._postprocess_tokenized_data_for_transformer(max_seq_len, pad_token_id, bos_token_id, eos_token_id)

        if train_val_test_split is None:
            train_val_test_split = [0.7, 0.15, 0.15]

        dataset_size = len(self.df)
        train_size = int(train_val_test_split[0] * dataset_size)
        val_size = int(train_val_test_split[1] * dataset_size)

        dataset = TransformerDataset(self.df)

        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, dataset_size - train_size - val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def _postprocess_tokenized_data_for_bert(self, max_len, random_token_start, random_token_end, pad_token_id, mask_token_id, cls_token_id, sep_token_id):
        column_list = self.df.columns.tolist()
        self.df.rename(columns={column_list[0]: "source", column_list[1]: "target"}, inplace=True)
        self.df["mask"] = self.df["target"]
        for idx in tqdm(range(len(self.df)), desc="Postprocessing tokenized data", total=len(self.df)):
            a_tokenized = self.df.at[idx, "source"]
            if random.random() > 0.5:
                b_tokenized = self.df.at[idx, "target"]
                isnext_token = 1
            else:
                i = random.randint(0, len(self.df) - 2)
                b_tokenized = self.df.at[i + 1 if i >= idx else i, "target"]
                isnext_token = 0

            # create tgt seq
            tgt_tokenized = [isnext_token] + a_tokenized + [sep_token_id] + b_tokenized
            target = tgt_tokenized[:max_len]
            target += [pad_token_id] * (max_len - len(target))

            # mask src
            length = len(a_tokenized) + len(b_tokenized)
            if length > max_len - 2:
                length = max_len - 2
            pred_list = random.sample(range(length), int(length * 0.15) if int(length * 0.15) > 0 else 1)

            for i in range(len(pred_list)):
                if pred_list[i] >= len(a_tokenized):
                    pred_list[i] += 2
                else:
                    pred_list[i] += 1

            src_tokenized = [cls_token_id] + a_tokenized + [sep_token_id] + b_tokenized

            mask_list = random.sample(pred_list, int(len(pred_list) * 0.9) if int(len(pred_list) * 0.9) > 0 else 1)
            for i in mask_list:
                src_tokenized[i] = mask_token_id

            random_list = random.sample(mask_list, int(len(mask_list) * (1 / 9)) if int(len(mask_list) * (1 / 9)) > 0 else 1)
            for i in random_list:
                src_tokenized[i] = random.randint(random_token_start, random_token_end)

            source = src_tokenized[:max_len]
            source += [pad_token_id] * (max_len - len(source))

            mask = []
            for i in range(len(source)):
                if i in pred_list:
                    mask.append(True)
                else:
                    mask.append(False)

            self.df.at[idx, "source"] = source
            self.df.at[idx, "target"] = target
            self.df.at[idx, "mask"] = mask

    def get_bert_dataloader(self, max_seq_len, batch_size, random_token_start, random_token_end, num_workers, pin_memory, pad_token_id=0, mask_token_id=1, cls_token_id=2, sep_token_id=3, train_val_test_split=None):
        self._postprocess_tokenized_data_for_bert(max_seq_len, random_token_start, random_token_end, pad_token_id, mask_token_id, cls_token_id, sep_token_id)

        if train_val_test_split is None:
            train_val_test_split = [0.7, 0.15, 0.15]

        dataset_size = len(self.df)
        train_size = int(train_val_test_split[0] * dataset_size)
        val_size = int(train_val_test_split[1] * dataset_size)

        dataset = BERTDataset(self.df)

        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, dataset_size - train_size - val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader
