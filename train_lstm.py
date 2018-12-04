import numpy as np

training_data = np.load('/datasets/home/32/032/sugao/CSE190/pa4/train.npy')

from dataloader import *
from models import *
from configs import cfg
import time

device = torch.device('cuda')
model = BeerLstmMind(cfg, True)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['L2_penalty'])
train_indices, valid_indices = train_valid_split(len(training_data), cfg['train_split'])
np.save('val3.npy', valid_indices)
print('ready')

for epoch in range(cfg['epochs']):
    tloader = DataLoader(training_data, train_indices, cfg)
    vloader = DataLoader(training_data, valid_indices, cfg)
    
    count = 0
    avg_loss = 0
    while tloader.has_next():
        #start = time.perf_counter()
        train, label = tloader.get_next()
        #end = time.perf_counter()
        #print('get train & label', end - start)
        
        model.zero_grad()
        
        #start = time.perf_counter()
        y = model(train)
        #end = time.perf_counter()
        #print('train', end - start)
        
        y = y.permute(0, 2, 1)
        
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()

        count += 1
        avg_loss += loss.item()
        if count % 100 == 0:
            print("count = %d, loss = %.5f" %(count, avg_loss / 100))
            save_model(model, 'models3/e' + str(epoch + 1) + 'b' + str(count) + '.pt')
            avg_loss = 0
        del train, label, y, loss
    # print('training loss:', avg_loss / count)
    
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
            del train, label, y, loss
    print('validation loss:', avg_loss / count)
    print('epoch finished:', epoch + 1)
    