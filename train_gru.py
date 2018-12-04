import numpy as np
from dataloader import *
from models import *
from configs import cfg
from torch.nn.functional import softmax
from nltk.translate import bleu_score
import time

training_data = np.load('/datasets/home/32/032/sugao/CSE190/pa4/train.npy')

cfg['hidden_dim'] = 128 # hidden dimension for LSTM
cfg['layers'] = 2 # number of layers of LSTM
cfg['dropout'] = 0.005 # dropout rate between two layers of LSTM; useful only when layers > 1; between 0 and 1
cfg['bidirectional'] = False # True or False; True means using a bidirectional LSTM
cfg['batch_size'] = 100 # batch size of input
cfg['train_split'] = 0.8
cfg['model_type'] = 'gru'
cfg['device'] = 'cuda'
cfg['output_dir'] = './reviews/'

models_dir = './models/2-4_5-5/'
val_dir = models_dir + 'val.npy'

device = torch.device('cuda')
model = BeerGruMind(cfg, True)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['L2_penalty'])
train_indices, valid_indices = train_valid_split(len(training_data), cfg['train_split'])
np.save(val_dir, np.array(valid_indices))
print(models_dir)

train_losses = []
valid_losses = []
for epoch in range(cfg['epochs']):
    tloader = DataLoader(training_data, train_indices, cfg)
    vloader = DataLoader(training_data, valid_indices, cfg)

    count = 0
    avg_loss = 0
    losss = []
    while tloader.has_next():
        train, label = tloader.get_next()

        model.zero_grad()
        
        y = model(train)

        y = y.permute(0, 2, 1)

        loss = criterion(y, label)
        loss.backward()
        optimizer.step()

        count += 1
        avg_loss += loss.item()
        if count % 150 == 0:
            print("count = %d, training loss = %.5f" %(count, avg_loss / 150) )
            save_model(model, models_dir + 'e%d_c%d_ls%.3f' % (epoch, count, avg_loss/150))
            losss.append(avg_loss)
            avg_loss = 0

        if epoch == 5 and count == 9500:
            break
    train_losses.append(losss)
    
    count = 0
    avg_loss = 0
    with torch.no_grad():
        while vloader.has_next():
            train, label = vloader.get_next()
            y = model(train)
            y = y.permute(0, 2, 1)
            loss = criterion(y, label)
            count += 1
            avg_loss += loss.item()
    print('validation loss:', avg_loss / count)
    print('epoch finished:', epoch + 1)
    valid_losses.append(avg_loss/count)

np.save(models+'val_loss.npy', np.array(valid_losses))
np.save(models+'train_loss.npy', np.array(train_losses))
