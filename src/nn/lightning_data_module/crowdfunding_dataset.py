import torch
from torch.utils.data import Dataset

import numpy as np

class CrowdfundingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len) -> None:
        self.len = len(df)
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row.text
        inputs = self.tokenizer.encode_plus(
            text,
            max_length = self.max_len,
            padding = "max_length",
            return_token_type_ids = True,
            truncation = True
        )
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        target = None if np.isnan(row.state) else torch.tensor(row.state, dtype=torch.long) 

        result = {}
        result["ids"] = row.id
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask
        
        if target is not None:
            result["target"] = target

        return result
    
    def __len__(self):
        return self.len