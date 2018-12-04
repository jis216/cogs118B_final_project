import numpy as np
import pandas as pd
import sys
from nltk.translate import bleu_score
import torch
from torch.nn.functional import softmax
import random

def load_data(fname):
    # TODO: From the csv file given by filename and return a pandas DataFrame of the read csv.
    return pd.read_csv(fname)
    

def process_train_data(row, styles):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).
    res = [0, 0]
    style = styles[row['beer/style']]
    res[0] = style
    res[1] = float(row['review/overall'])
    for j, char in enumerate(row['review/text']):
        code = char2oh(char)
        if code == -1:
            continue
        res.append(code)
    return res


def train_valid_split(data, labels):
    # TODO: Takes in train data and labels as numpy array (or a torch Tensor/ Variable) and
    # splits it into training and validation data.
    indices = np.arange(len(data))
    random.shuffle(indices)
    split = int(len(data) * 0.8)
    
    return data[:split],data[split:],labels[:split],labels[split:]
    
    
def process_test_data(data):
    # TODO: Takes in pandas DataFrame and returns a numpy array (or a torch Tensor/ Variable)
    # that has all input features. Note that test data does not contain any review so you don't
    # have to worry about one hot encoding the data.
    unistyle = data['beer/style'].unique()
    styles = dict(zip(unistyle, range(len(unistyle))))
    
    all_features = []
    for index, row in data.iterrows():
        feat = [0] * 3
        style = styles[row['beer/style']]
        feat[0] = style
        feat[1] = float(row['review/overall'])
        feat[2] = 97
        all_features.append(feat)
        
    return all_features

    
def pad_data(orig_data):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.

    # Implemented in dataloader.py

    raise NotImplementedError
    

def train(model, X_train, y_train, X_valid, y_valid, cfg, write):
    # TODO: Train the model!

    # Implemented in train_gru.py and train_lstm.py

    raise NotImplementedError
    
    
def generate(model, X_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    device = torch.device(cfg['device'])
    carry = torch.zeros(len(X_test), 1, 204, dtype=torch.float, device=device)
    carry[[x for x in range(len(X_test))], 0, X_test[:, 0]] = 1
    carry[:, 0, 104] = torch.tensor(X_test[:, 1], device=device)
    carry[:, 0, 202] = 1
#     for i in range(len(X_test)):
#         assert int(carry[i, 0, int(X_test[i, 0])].item()) == 1
#         assert carry[i, 0, 104].item() == X_test[i, 1]
#         assert int(carry[i, 0, 202].item()) == 1
#         assert int((carry[i, 0, :104].sum() + carry[i, 0, 105:].sum()).item()) == 2
        
    all_txt = torch.empty(len(X_test), cfg['max_len'], dtype=torch.long, device=device)
    model.reset_hidden()
    
    for i in range(cfg['max_len']):
        y = model(carry)
        y /= cfg['gen_temp']
        y = torch.nn.functional.softmax(y, dim=2)

        y[:, 0, 95] = 0
        y[:, 0, 97] = 0

        vals = torch.multinomial(y[:, 0, :], 1)
        all_txt[:, i] = vals[:, 0]
        carry[:, 0, 105:] = 0
        carry[[x for x in range(len(X_test))], 0, vals[:, 0] + 105] = 1
        
    return all_txt


def tensor2strlist(m):
    all_txt = []
    size = m.size()
    for i in range(size[0]):
        txt = []
        buf = ''
        for j in range(size[1]):
            val = m[i, j].item()
            if val == 98:
                break
            if val == 0:
                txt.append(buf)
                buf = ''
                continue
            buf += oh2char(val)
        txt.append(buf)
        all_txt.append(txt)
    return all_txt
    
    
def calc_score(df, indices, all_txt, smooth=None):
    references = [d.split() for d in df['review/text'].values[indices]]
    #hypotheses = [t.split() for t in all_txt]
    score = bleu_score.corpus_bleu(references, all_txt, smoothing_function=smooth)
    return score
    
    
def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    file = open(fname, 'a')
    for i in range(len(outputs)):
        file.write(' '.join(outputs[i]) + '\n')
    file.close()
    
    
def char2oh(char):
    char = ord(char)
    if char >= 32 and char <= 126:
        return char - 32
    elif char == 10: # newline
        return 95
    elif char == 9:  # tab
        return 96
    else:
        print('char out of range:', char)
        return -1
        
        
def oh2char(oh):
    if type(oh) is int:
        val = oh
    else:
        val = torch.nonzero(oh[105:])
        
    if val == 95:
        return chr(10)
    elif val == 96:
        return chr(9)
    elif val > 96:
        return ' '
    else:
        return chr(32 + val)
    
    
if __name__ == '__main__':
    train_data_fname = '/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Train.csv'
    test_data_fname = '/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Test.csv'
    
    res = []
    if len(sys.argv) > 1 and sys.argv[1] == 'gen-train':
        train_data = load_data(train_data_fname) # Generating the pandas DataFrame
        unistyle = train_data['beer/style'].unique()
        styles = dict(zip(unistyle, range(len(unistyle))))
        limit = float('inf')
        if len(sys.argv) > 2:
            limit = int(sys.argv[2])
        
        for index, row in train_data.iterrows():
            if type(row['review/text']) is not str:
                print('drop missing data', index)
                continue
            arr = process_train_data(row, styles)
            res.append(arr)
            if (index + 1) % 5000 == 0:
                print(index + 1)
            if index + 1 == limit:
                break
        np.save('train.npy', res)
        
    if len(sys.argv) > 1 and sys.argv[1] == 'gen-test':
        test_data = load_data(test_data_fname) # Generating the pandas DataFrame
        test_data.dropna(subset=['review/overall', 'beer/style'],how='any',inplace = True)
        
        res = process_test_data(test_data)
        np.save('test.npy', res)
            
    
