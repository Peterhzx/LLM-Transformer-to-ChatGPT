import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self, df):
        super(TransformerDataset, self).__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.df.at[idx, "src_input"], dtype=torch.long)
        decoder_input_tensor = torch.tensor(self.df.at[idx, "decoder_input"], dtype=torch.long)
        target_tensor = torch.tensor(self.df.at[idx, "target"], dtype=torch.long)

        return src_tensor, decoder_input_tensor, target_tensor


class BERTDataset(Dataset):
    def __init__(self, df):
        super(BERTDataset, self).__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # improvement: Instead of fixed-length sequences, consider dynamic batching with a custom collate_fn
        source_tensor = torch.tensor(self.df.at[idx, "source"], dtype=torch.long)
        target_tensor = torch.tensor(self.df.at[idx, "target"], dtype=torch.long)
        mask_tensor = torch.tensor(self.df.at[idx, "mask"], dtype=torch.long)

        return source_tensor, target_tensor, mask_tensor
