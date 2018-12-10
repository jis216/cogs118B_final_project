import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as func
    
    
class GRU_Score(nn.Module):
    def __init__(self, cfg, reset):
        super(GRU_Score, self).__init__()
        self.embed_true = cfg['embed']
        if self.embed_true:
            self.embedding = nn.Embedding(cfg['vocab_dim'], cfg['input_dim'])
        
        self.embed_linear = nn.Linear(cfg['input_dim'], cfg['input_dim'])
        torch_init.xavier_normal_(self.embed_linear.weight)
        
        self.gru = nn.GRU(cfg['input_dim'], cfg['hidden_dim'], cfg['layers'],
                            batch_first=True, dropout=cfg['dropout'], bidirectional=cfg['bidirectional'])
        
        if cfg['bidirectional']:
            self.fc1 = nn.Linear(2*cfg['hidden_dim']*cfg['layers'], 64)
        else:
            self.fc1 = nn.Linear(cfg['hidden_dim']*cfg['layers'], 64)
        self.fc1_normed = nn.BatchNorm1d(64)
        torch_init.xavier_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(64, 32)
        self.fc2_normed = nn.BatchNorm1d(32)
        torch_init.xavier_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(32, cfg['output_dim'])
        torch_init.xavier_normal_(self.fc3.weight)
        
        self.hidden = torch.zeros(cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float, 
                                  device=torch.device(cfg['device']))
        
        self.reset = reset

        
    def reset_hidden(self):
        self.hidden[:] = 0
        
        
    def forward(self, sequence):
        batch = self.embed_linear(sequence)
        sequence = func.relu(batch)
        
        if self.reset:
            gru_out, self.hidden = self.gru(sequence)
        else:
            gru_out, self.hidden = self.gru(sequence, self.hidden)
        
        batch = self.hidden
        dim = batch.size()
        batch = batch.view(-1, dim[0]*dim[2])
        
        fc1_out = func.relu(self.fc1_normed(self.fc1(batch)))
        fc2_out = func.relu(self.fc2_normed(self.fc2(fc1_out)))
        fc3_out = self.fc3(func.relu(fc2_out))
        return fc3_out
    
class LSTM_Score(nn.Module):
    def __init__(self, cfg, reset, embed_dict=None):
        super(LSTM_Score, self).__init__()
        self.embed_true = cfg['embed']
        if self.embed_true:
            self.embedding = nn.Embedding(cfg['vocab_dim'], cfg['input_dim'])
        
        self.embed_linear = nn.Linear(cfg['input_dim'], cfg['input_dim'])
        torch_init.xavier_normal_(self.embed_linear.weight)
            
        self.lstm = nn.LSTM(cfg['input_dim'], cfg['hidden_dim'], cfg['layers'],
                            batch_first=True, dropout=cfg['dropout'], bidirectional=cfg['bidirectional'])
        
        if cfg['bidirectional']:
            self.fc1 = nn.Linear(2*cfg['hidden_dim']*cfg['layers'], 64)
        else:
            self.fc1 = nn.Linear(cfg['hidden_dim']*cfg['layers'], 64)
        self.fc1_normed = nn.BatchNorm1d(64)
        torch_init.xavier_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(64, 32)
        self.fc2_normed = nn.BatchNorm1d(32)
        torch_init.xavier_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(32, cfg['output_dim'])
        torch_init.xavier_normal_(self.fc3.weight)
        
        self.hidden = torch.zeros(cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float, 
                                  device=torch.device(cfg['device']))
        self.cell = torch.zeros(cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float,
                                device=torch.device(cfg['device']))
        self.reset = reset

        
    def reset_hidden(self):
        self.hidden[:] = 0
        self.cell[:] = 0
        
        
    def forward(self, sequence):
        batch = self.embed_linear(sequence)
        #print(sequence.size())
        #print(batch.size())
        sequence = func.relu(batch)
        
        if self.reset:
            lstm_out, (self.hidden, self.cell) = self.lstm(sequence)
        else:
            lstm_out, (self.hidden, self.cell) = self.lstm(sequence, (self.hidden, self.cell))
        #print(self.hidden.size())
        #print(lstm_out.size())
        
        #batch = self.hidden.permute(1, 0, 2)
        batch = self.hidden
        #print(batch.size())
        dim = batch.size()
        batch = batch.view(-1, dim[0]*dim[2])
        #print(batch.size())
        fc1_out = func.relu(self.fc1_normed(self.fc1(batch)))
        fc2_out = func.relu(self.fc2_normed(self.fc2(fc1_out)))
        fc3_out = self.fc3(func.relu(fc2_out))
        return fc3_out
    
def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(ctor, path, cfg):
    model = ctor(cfg, False)
    model = model.to(torch.device(cfg['device']))
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
    
    