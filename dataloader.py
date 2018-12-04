import random
import torch
import numpy as np


def train_valid_split(N, split_ratio):
    indices = list(range(N))
    random.shuffle(indices)
    split = round(N * split_ratio)
    
    return indices[:split], indices[split:]
    
    
class DataLoader:
    def __init__(self, data, indices, cfg):
        self.data = data
        self.rest = indices[:]
        random.shuffle(self.rest)
        self.batch_size = cfg['batch_size']
        self.device = torch.device(cfg['device'])

        
    def get_next(self):
        # get and delete last batch_size of indices
        batch_index = self.rest[-self.batch_size:]
        del self.rest[-self.batch_size:]

        batch = []

        # find the longest sequence in this batch
        seq = 0

        styles = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        ratings = torch.zeros(self.batch_size, 1, dtype=torch.float, device=self.device)
        char_codes = []
        
        for i, index in enumerate(batch_index):
            row = self.data[index]
            
            l = len(row[2:])
            if l > seq:
                seq = l
            styles[i] = row[0]
            ratings[i, 0] = row[1]
            
            char_codes.append([97] + row[2:])

            batch.append(row)

        seq = seq + 1

        if len(char_codes) == 1:
            char_codes += [98]
        else:
            for c in char_codes:
                c += (seq - len(c) + 1) * [98]
                
        char_codes = torch.tensor(char_codes, dtype=torch.long, device=self.device)
        
        train = torch.zeros(self.batch_size, seq, 204, dtype=torch.float, device=self.device)
        label = torch.zeros(self.batch_size, seq, dtype=torch.long, device=self.device)
        
        n_indices = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.device)
        batch_arange = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        n_indices[batch_arange,0] = batch_arange
        
        # set style
        train[batch_arange, :, styles] = 1
        
        # set rating
        train[batch_arange, :, 104] = ratings

        seq_arange = torch.arange(seq, dtype=torch.long, device=self.device)
        train[n_indices, seq_arange, char_codes[:,:-1]+105] = 1 
        label[n_indices, seq_arange] = char_codes[:,1:]

        return train, label
    
    
    def has_next(self):
        return len(self.rest) >= self.batch_size
        
