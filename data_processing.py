import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataProcessor:
    def __init__(self, df, train_ratio=0.8):
        self.df = df
        self.train_ratio = train_ratio
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def preprocess(self):
        split_index = int(len(self.df) * self.train_ratio)
        train_df = self.df[:split_index]
        test_df = self.df[split_index:]

        train_norm = self.scaler.fit_transform(train_df)
        test_norm = self.scaler.transform(test_df)

        train_norm = torch.FloatTensor(train_norm).view(-1)
        test_norm = torch.FloatTensor(test_norm).view(-1)
        
        return train_norm, test_norm
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(np.array(data).reshape(-1, 1))

    @staticmethod
    def input_data(seq, window_size):
        out = []
        L = len(seq)
        for i in range(L - window_size):
            window = seq[i:i + window_size]
            label = seq[i + window_size:i + window_size + 1]
            out.append((window, label))
        return out
