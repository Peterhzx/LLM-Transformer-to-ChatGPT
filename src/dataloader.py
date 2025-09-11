import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets import TransformerDataset, BERTDataset


class Dataloader:
    def __init__(self, data_file, allowed_chars="", nrows=None, mode="local"):
        df_path = data_file
        if mode == "sagemaker":
            parent = os.environ.get('SM_CHANNEL_TRAIN', "/opt/ml/input/data/train")
            df_path = os.path.join(parent, df_path)

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

    def get_df(self, sample_size=None, n=None):
        return self.df.sample(n=n, frac=sample_size)

    def get_transformer_dataloader(self, max_seq_len, batch_size, num_workers=0, pin_memory=False, pad_token_id=0, bos_token_id=1, eos_token_id=2, seed=42, train_val_test_split=None):
        if train_val_test_split is None:
            train_val_test_split = [0.7, 0.15, 0.15]

        dataset_size = len(self.df)
        train_size = int(train_val_test_split[0] * dataset_size)
        val_size = int(train_val_test_split[1] * dataset_size)

        dataset = TransformerDataset(self.df, max_seq_len, pad_token_id, bos_token_id, eos_token_id)

        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, dataset_size - train_size - val_size], generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader

    def get_bert_dataloader(self, max_seq_len, batch_size, random_token_start, random_token_end, num_workers=0, pin_memory=False, pad_token_id=0, mask_token_id=1, cls_token_id=2, sep_token_id=3, seed=42, train_val_test_split=None):
        if train_val_test_split is None:
            train_val_test_split = [0.7, 0.15, 0.15]

        dataset_size = len(self.df)
        train_size = int(train_val_test_split[0] * dataset_size)
        val_size = int(train_val_test_split[1] * dataset_size)

        dataset = BERTDataset(self.df, max_seq_len, random_token_start, random_token_end, pad_token_id, mask_token_id, cls_token_id, sep_token_id)

        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, dataset_size - train_size - val_size], generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader
