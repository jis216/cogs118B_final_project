import random
import torch
import numpy as np
from string import punctuation
from preprocess import *
import torchwordemb

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
    sequence_feats = np.zeros((len(sequence), 301))
    #sequence_feats[:,0] = float(score)
    
    # SOE is [-1, 300 * -1]
    # sequence_feats[0,0] = -1
    
    for ind, word in enumerate(sequence):
        if word not in word_dict:
            word = ''.join([c for c in word if not c.isdigit()])
        if word not in word_dict:
            sequence_feats[ind, 0] = -1
            for i,c in enumerate(word):
                sequence_feats[ind, 1 + i] = (ord(c) - 32) / (126-32) * 2 - 1
        else:
            sequence_feats[ind, 0] = 1
            #sequence_feats[ind+1, 1:] = torch.FloatTensor(word_dict[word].tolist(),device=device)
            sequence_feats[ind, 1:] = word_dict[word]

               
    # EOE is [-1, 300 * 0]
    #sequence_feats[-1,0] = -1
    
    #print(np.asarray(essay_feats)[:,0])
    #print(essay_feats[len(essay_feats)-1])
    
    return sequence_feats.tolist()  

class GloveVocabBuilder(object) :

    def __init__(self, path_glove):
        self.vec = None
        self.vocab = None
        self.path_glove = path_glove

    def get_word_index(self, padding_marker='__PADDING__', unknown_marker='__UNK__',):
        _vocab, _vec = torchwordemb.load_glove_text(self.path_glove)
        vocab = {padding_marker:0, unknown_marker:1}
        for tkn, indx in _vocab.items():
            vocab[tkn] = indx + 2
        vec_2 = torch.zeros((2, _vec.size(1)))
        vec_2[1].normal_()
        self.vec = torch.cat((vec_2, _vec))
        self.vocab = vocab
        return self.vocab, self.vec
    
def build_embed_tensor(glove_p):
    v_builder, d_word_index, embed = None, None, None
    v_builder = GloveVocabBuilder(path_glove=glove_p)
    d_word_index, embed = v_builder.get_word_index()
    args.embedding_size = embed.size(1)
     
import torch
import pandas as pd
import numpy as np

import util as ut

class TextClassDataLoader(object):

    def __init__(self, path_file, word_to_index, batch_size=32):
        """
        Args:
            path_file:
            word_to_index:
            batch_size:
        """

        self.batch_size = batch_size
        self.word_to_index = word_to_index

        # read file
        df = pd.read_csv(path_file, delimiter='\t')
        df['body'] = df['body'].apply(ut._tokenize)
        df['body'] = df['body'].apply(self.generate_indexifyer())
        self.samples = df.values.tolist()

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = int(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        self._shuffle_indices()

        self.report()

    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[1]))
        return length

    def generate_indexifyer(self):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index['__UNK__'])
            return indices

        return indexify

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(map(len, string))

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        label = torch.LongTensor(label)
        label = label[perm_idx]

        return seq_tensor, label, seq_lengths

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))