import numpy as np
import pandas as pd
import sys
from nltk.translate import bleu_score
import torch
from torch.nn.functional import softmax
import random
import string
from string import punctuation
import pickle

def load_data(fname):
    # TODO: From the csv file given by filename and return a pandas DataFrame of the read csv.
    df = pd.read_table(fname, encoding = "ISO-8859-1")
    return df
    
def process_train_data(text, score, word_dict, device):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).    
    lowered = ''.join([(' '+c+' ') if c in punctuation else c for c in text.lower() ])
    sequence = lowered.split()
    
    scores = np.repeat(float(score),len(sequence)+2).reshape((len(sequence)+2,1)).tolist()
    
    # SOS is [1, 300 * 0]
    essay_feats = torch.zeros(1, 301, dtype=torch.long, device=device)
    essay_feats[0] = -1

    for word in sequence:
        if word not in word_dict:
            word = ''.join([c for c in word if not c.isdigit()])
        if word not in word_dict:
            feat  = [0] * 300
            for i,c in enumerate(word):
                feat[ord(c)] = (i+1) * 0.05 - 0.5
            code = [1] + feat
        else:
            code = [0] + word_dict[word].tolist()
        essay_feats.append(code)
        
    # EOS is [1, 300 * 1]
    essay_feats.append([-1]+ [1] * 300 )
    
    return np.hstack((scores,essay_feats))
    
    
def save_dict(dictionary, path ):
    with open(path, 'wb') as fp:
        pickle.dump(dictionary, fp)

def load_dict(path ):
    with open(path, 'rb') as f:
        return pickle.load(f)
    