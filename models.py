import torch
import torch.nn as nn
import torch.nn.init as torch_init

class BeerLstmMind(nn.Module):
    def __init__(self, cfg, reset):
        super(BeerLstmMind, self).__init__()
        self.lstm = nn.LSTM(cfg['input_dim'], cfg['hidden_dim'], cfg['layers'],
                            batch_first=True, dropout=cfg['dropout'], bidirectional=cfg['bidirectional'])
        
        self.fc = nn.Linear(cfg['hidden_dim'], cfg['output_dim'])
        torch_init.xavier_normal_(self.fc.weight)
        
        self.hidden = torch.zeros(cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float, 
                                  device=torch.device(cfg['device']))
        self.cell = torch.zeros(cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float,
                                device=torch.device(cfg['device']))
        self.reset = reset

        
    def reset_hidden(self):
        self.hidden[:] = 0
        self.cell[:] = 0
        
        
    def forward(self, sequence):
        if self.reset:
            lstm_out, (self.hidden, self.cell) = self.lstm(sequence)
        else:
            lstm_out, (self.hidden, self.cell) = self.lstm(sequence, (self.hidden, self.cell))
        fc_out = self.fc(lstm_out)
        return fc_out
    
    
class BeerGruMind(nn.Module):
    def __init__(self, cfg, reset):
        super(BeerGruMind, self).__init__()
        self.gru = nn.GRU(cfg['input_dim'], cfg['hidden_dim'], cfg['layers'],
                          batch_first=True, dropout=cfg['dropout'], bidirectional=cfg['bidirectional'])
        
        self.fc = nn.Linear(cfg['hidden_dim'], cfg['output_dim'])
        torch_init.xavier_normal_(self.fc.weight)
        
        self.hidden = torch.zeros(cfg['layers'], cfg['batch_size'], cfg['hidden_dim'], dtype=torch.float, 
                                  device=torch.device(cfg['device']))
        self.reset = reset

        
    def reset_hidden(self):
        self.hidden[:] = 0
        
        
    def forward(self, sequence):
        if self.reset:
            gru_out, self.hidden = self.gru(sequence)
        else:
            gru_out, self.hidden = self.gru(sequence, self.hidden)
        fc_out = self.fc(gru_out)
        return fc_out

    
def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(ctor, path, cfg):
    model = ctor(cfg, False)
    model = model.to(torch.device(cfg['device']))
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
    
    