from datasets import MyDataSet
from models.fm import FM
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from sklearn import metrics

LABEL = 'actual_capital'
DATA_PATH = 'data/final_table.csv'
CHECKPOINT_PATH = 'stats/fm/'
EPOCH_NUM = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.01
K = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = FM(n=18, k=5)
train_dataset = MyDataSet(DATA_PATH, LABEL, 1)
test_dataset = MyDataSet(DATA_PATH, LABEL, 0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = F.binary_cross_entropy
start_epoch = 0

model.to(device=device)

checkpoint_path = os.path.join(CHECKPOINT_PATH, LABEL)
if os.path.exists(checkpoint_path) == False: os.mkdir(checkpoint_path)
checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.tar')
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Checkpoint File Loaded')
else: print('No Trained Model')




def test():
    all_y = {}
    all_pred = {}
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            x, y = data
            x, y = x.to(torch.float32), y.to(torch.float32)
            x, y = x.to(device), y.to(device)
            pred = model(x)

            if batch_idx == 0:
                all_y = y
                all_pred = pred
            else:
                all_y = torch.cat((all_y, y), axis=0)
                all_pred = torch.cat((all_pred, pred), axis=0)
    AUC = metrics.roc_auc_score(all_y, all_pred)
    print('AUC: ', AUC)
             



if __name__ == '__main__':
    test()
