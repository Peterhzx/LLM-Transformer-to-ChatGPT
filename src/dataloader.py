import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Dataloader:
    def __init__(self, config):
        df_path = config["data_path"]
        allowed_chars = config.get("allowed_chars", "")
        nrows = config.get("nrows", None)

        if df_path.endswith(".csv"):
            self.df = pd.read_csv(df_path, nrows=nrows)
        elif df_path.endswith(".xlsx"):
            self.df = pd.read_excel(df_path, nrows=nrows)
        elif df_path.endswith(".json"):
            self.df = pd.read_json(df_path, nrows=nrows)
        else:
            raise ValueError("file formate should be: .csv .xlsx .json")

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

    def get_df(self, sample_size=1000000):
        return self.df.sample(n=sample_size)

    def get_transformer_dataloader(self, max_seq_len, batch_size, train_val_test_split=None):
        if train_val_test_split is None:
            train_val_test_split = [0.7, 0.15, 0.15]

        dataset_size = len(self.df)
        train_size = int(train_val_test_split[0] * dataset_size)
        val_size = int(train_val_test_split[1] * dataset_size)

        train_set = TransformerDataset(self.df.iloc[:train_size], max_seq_len)
        val_set = TransformerDataset(self.df.iloc[train_size:(train_size + val_size)], max_seq_len)
        test_set = TransformerDataset(self.df.iloc[(train_size + val_size):dataset_size], max_seq_len)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


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

    def __getitem__(self,
                    idx):  # improvement: Instead of fixed-length sequences, consider dynamic batching with a custom collate_fn
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
