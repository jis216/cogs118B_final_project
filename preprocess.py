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

def load_all_training_set(train_path):
    training_data = load_data(train_path)
    training_data.dropna(subset=['essay_set','domain1_score', 'essay'],how='any',inplace = True)
    training_data = training_data[['essay_set','domain1_score', 'domain2_score','essay']]
    training_data.domain1_score = training_data.domain1_score.astype('float64')
    
    return training_data

def load_training_set(train_path, set_id):
    training_data = load_data(train_path)
    training_data.dropna(subset=['essay_set','domain1_score', 'essay'],how='any',inplace = True)
    training_data = training_data[['essay_set','domain1_score', 'domain2_score','essay']]
    training_data = training_data [training_data.essay_set == set_id]
    training_data.domain1_score = training_data.domain1_score.astype('float64')
    
    return training_data

def load_all_validation_set(valid_path, valid_label_path):
    
    valid_data = load_data(valid_path)
    valid_label = pd.read_csv(valid_label_path)
    
    label_dict = dict(zip(valid_label.prediction_id, valid_label.predicted_score))
    for (i,row) in valid_data.iterrows():
        valid_data.at[i,'domain1_predictionid'] = label_dict[row['domain1_predictionid']]
    valid_data = valid_data[['essay_set','essay','domain1_predictionid']]
    valid_data = valid_data.rename(index=str, columns={'domain1_predictionid':'domain1_score'})
    valid_data.domain1_score = valid_data.domain1_score.astype('float64')
    
    return valid_data

def load_validation_set(valid_path, valid_label_path, set_id):
    
    valid_data = load_data(valid_path)
    valid_label = pd.read_csv(valid_label_path)
    
    label_dict = dict(zip(valid_label.prediction_id, valid_label.predicted_score))
    for (i,row) in valid_data.iterrows():
        valid_data.at[i,'domain1_predictionid'] = label_dict[row['domain1_predictionid']]
    valid_data = valid_data[['essay_set','essay','domain1_predictionid']]
    valid_data = valid_data [valid_data.essay_set == set_id]
    valid_data = valid_data.rename(index=str, columns={'domain1_predictionid':'domain1_score'})
    valid_data.domain1_score = valid_data.domain1_score.astype('float64')
    
    return valid_data

def generate_word_idx(data):
    word2ind = {}
    for text in data.essay:
        lowered = ''.join([(' '+c+' ') if c in punctuation else c for c in text.lower() ])
        sequence = lowered.split()
        for word in sequence:
            if word not in word2ind:
                word2ind[word] = len(word2ind)
                
    return word2ind

def process_scores(data, score_domain):
    
    for (i,row) in data.iterrows():
        col = score_domain
        if row['essay_set'] == 1:
            data.at[i, col] = (row[score_domain] - 2)/10
        elif row['essay_set'] == 2:
            data.at[i, col] =(row[score_domain] - 1)/5
        elif row['essay_set'] == 3 or row['essay_set'] == 4:
            data.at[i, col] =row[score_domain]/3.0
            
        elif row['essay_set'] == 5 or row['essay_set'] == 6:
            data.at[i, col]= row[score_domain]/4.0
            
        elif row['essay_set'] == 7:
            data.at[i, col] =row[score_domain]/30
            
        elif row['essay_set'] == 8:
            data.at[i, col] =row[score_domain]/60
    return data
    
    