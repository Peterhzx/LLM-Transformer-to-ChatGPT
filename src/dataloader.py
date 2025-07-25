import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import TransformerDataset


class Dataloader:
    def __init__(self, config):
        df_path = config["data_path"]
        allowed_chars = config.get("allowed_chars", "")
        nrows = config.get("nrows", None)

        print("Loading data...")
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

    def get_df(self, sample_size=0.1):
        return self.df.sample(frac=sample_size)

    def tokenize_df(self, tokenizer, params):
        self.df = tokenizer.tokenize(self.df, params)

    def get_transformer_dataloader(self, tokenizer, max_seq_len, batch_size, train_val_test_split=None):
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
