import random
import torch
import numpy as np
from string import punctuation


def train_valid_split(N, split_ratio):
    indices = list(range(N))
    random.shuffle(indices)
    split = round(N * split_ratio)
    
    return indices[:split], indices[split:]

def process_train_data(text, word_dict,device):
            
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).    
    lowered = ''.join([(' '+c+' ') if c in punctuation else c for c in text.lower() ])
    sequence = lowered.split()
        
    #sequence_feats = torch.zeros(len(sequence)+2, 301, dtype=torch.float, device=device)
    sequence_feats = np.zeros((len(sequence)+1, 301))
    #sequence_feats[:,0] = float(score)
    
    # SOE is [-1, 300 * -1]
    # sequence_feats[0,0] = -1
    
    for ind, word in enumerate(sequence):
        if word not in word_dict:
            word = ''.join([c for c in word if not c.isdigit()])
        if word not in word_dict:
            sequence_feats[ind, 0] = 1
            for i,c in enumerate(word):
                sequence_feats[ind, 1 + ord(c)] += (i+1) * 0.05 - 0.5
        else:
            sequence_feats[ind, 0] = 0
            #sequence_feats[ind+1, 1:] = torch.FloatTensor(word_dict[word].tolist(),device=device)
            sequence_feats[ind, 1:] = word_dict[word]

               
    # EOE is [-1, 300 * 0]
    sequence_feats[-1,0] = -1
    
    #print(np.asarray(essay_feats)[:,0])
    #print(essay_feats[len(essay_feats)-1])
    
    return sequence_feats.tolist()  
    
class DataLoader:
    def __init__(self, data, indices, cfg, word_dict):
        self.data = data
        self.rest = indices[:]
        random.shuffle(self.rest)
        self.batch_size = cfg['batch_size']
        self.device = torch.device(cfg['device'])
        self.word_dict = word_dict
        self.cfg = cfg
        
    def get_next(self):
        # get and delete last batch_size of indices
        batch_index = self.rest[-self.batch_size:]
        del self.rest[-self.batch_size:]
                
        scores = self.data['domain1_score']
        essays = self.data['essay']

        # find the longest sequence in this batch
        seq = 0
        essay_codes = []
        
        for i, index in enumerate(batch_index):
            essay = (self.data['essay'])[index]
            text = process_train_data(essay, self.word_dict, 'cuda')
            essay_codes.append(text)
            
            l = len(text)

            if l > seq:
                seq = l
                
        eos = [0] * 301
        eos[0] = -1
        
        for j,es in enumerate(essay_codes):
            #print()
            essay_codes[j] += (seq - len(es)) * [eos]

        essay_codes = torch.tensor(essay_codes, dtype=torch.float, device=self.device)
        
        train = torch.zeros(self.batch_size, seq, self.cfg['input_dim'], dtype=torch.float, device=self.device)
        label = torch.zeros(self.batch_size, 1, dtype=torch.float, device=self.device)
        
        n_indices = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.device)
        batch_arange = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        n_indices[batch_arange,0] = batch_arange
        
        train[batch_arange] = essay_codes
        
        scores = (self.data['domain1_score'])[batch_index]
        scores = [[s] for s in scores]
        
        label[n_indices, 0] = torch.tensor(scores,dtype=torch.float, device = self.device)

        del scores
        return train, label
    
    
    def has_next(self):
        return len(self.rest) >= self.batch_size
        
