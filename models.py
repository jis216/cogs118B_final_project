import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    
class GRU_Score(nn.Module):
    def __init__(self, cfg, reset, use_last = False):
        super(GRU_Score, self).__init__()
        
        self.embed_linear = nn.Linear(cfg['input_dim'], cfg['input_dim'])
        torch_init.xavier_normal_(self.embed_linear.weight)
        
        self.gru = nn.GRU(cfg['input_dim'], cfg['hidden_dim'], cfg['layers'],
                            batch_first=True, dropout=cfg['dropout'], bidirectional=cfg['bidirectional'])
        
        self.bn = nn.BatchNorm1d(cfg['hidden_dim']*2)
        self.fc1 = nn.Linear(cfg['hidden_dim']*2, 1)
        torch_init.xavier_normal_(self.fc1.weight)
        
        self.reset = reset
        self.batch_size = cfg['batch_size']
        self.use_last = use_last
        
        self.hidden = torch.zeros(2*cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float, 
                                  device=torch.device(cfg['device']))
        
        self.reset = reset
        
        self.batch_size = cfg['batch_size']
        self.use_last = use_last
        
    def reset_hidden(self):
        self.hidden[:] = 0
        
        
    def forward(self, sequence, seq_lengths):
        batch = self.embed_linear(sequence)
        
        if self.reset:
            gru_out, self.hidden = self.gru(batch)
        else:
            gru_out, self.hidden = self.gru(batch, self.hidden)
            
        row_indices = torch.arange(0, sequence.size(0)).long().cuda()
        col_indices = (seq_lengths - 1).cuda()
        
        
        if self.use_last:
            last_tensor= gru_out[row_indices, col_indices, :]
        else:
            # use mean
            last_tensor = gru_out[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)
        
        
        fc_input = self.bn(last_tensor)
        out = func.sigmoid(self.fc1(fc_input))
        return out
    
class LSTM_Score(nn.Module):
    def __init__(self, cfg, reset, use_last = False):
        super(LSTM_Score, self).__init__()
        
        self.embed_linear = nn.Linear(cfg['input_dim'], cfg['input_dim'])
        torch_init.xavier_normal_(self.embed_linear.weight)
           
        self.lstm = nn.LSTM(cfg['input_dim'], cfg['hidden_dim'], cfg['layers'],
                            batch_first=True, dropout=cfg['dropout'], bidirectional=cfg['bidirectional'])
        
        self.bn = nn.BatchNorm1d(cfg['hidden_dim']*2)
        self.fc1 = nn.Linear(cfg['hidden_dim']*2, 1)
        torch_init.xavier_normal_(self.fc1.weight)
        
        
        self.hidden = torch.zeros(2*cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float, 
                                  device=torch.device(cfg['device']))
        self.cell = torch.zeros(2*cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float,
                                device=torch.device(cfg['device']))
        self.reset = reset
        self.batch_size = cfg['batch_size']
        self.use_last = use_last
        
    def reset_hidden(self):
        self.hidden[:] = 0
        self.cell[:] = 0
        self.time = 0
        
    def forward(self, sequence, seq_lengths):
        
        batch = self.embed_linear(sequence)
        
        if self.reset:
            lstm_out, (self.hidden, self.cell) = self.lstm(batch)
        else:
            lstm_out, (self.hidden, self.cell) = self.lstm(batch, (self.hidden, self.cell))
            
        row_indices = torch.arange(0, sequence.size(0)).long().cuda()
        col_indices = (seq_lengths - 1).cuda()
        
        
        if self.use_last:
            last_tensor= lstm_out[row_indices, col_indices, :]
        else:
            # use mean
            last_tensor = lstm_out[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)
        
        
        fc_input = self.bn(last_tensor)
        out = func.sigmoid(self.fc1(fc_input))
        return out

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(ctor, path, cfg):
    model = ctor(cfg, False)
    model = model.to(torch.device(cfg['device']))
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
    
    