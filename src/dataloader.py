import pandas as pd
from tqdm import tqdm


class Dataloader:
    def __init__(self, df_path, allowed_chars=""):
        if df_path.endwith("v"):
            self.df = pd.read_csv(df_path)
        elif df_path.endwith("x"):
            self.df = pd.read_excel(df_path)
        elif df_path.endwith("n"):
            self.df = pd.read_json(df_path)
        else:
            raise ValueError("file formate should be: .csv .xlsx .json")

        if type(allowed_chars) == list:
            code_points = sorted(set(allowed_chars))
            self.allowed_chars = ''.join(chr(cp) for cp in code_points)
        elif type(allowed_chars) == str:
            self.allowed_chars = allowed_chars
        else:
            raise ValueError("allowed_chars accept only list and str")

    def clean_data(self):
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
