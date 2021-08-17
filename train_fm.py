from datasets import MyDataSet
from models.fm import FM
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from sklearn import metrics

LABEL = 'punishment'
DATA_PATH = 'data/final_table.csv'
CHECKPOINT_PATH = 'stats/fm/'
NEW_MODEL_FLAG = True   ### Whether to train a new model instead of loading checkpoint
EPOCH_NUM = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = FM(n=18, k=5)
train_dataset = MyDataSet(DATA_PATH, LABEL, 1)
test_dataset = MyDataSet(DATA_PATH, LABEL, 0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
criterion = F.binary_cross_entropy
start_epoch = 0

model.to(device=device)

checkpoint_path = os.path.join(CHECKPOINT_PATH, LABEL)
if os.path.exists(checkpoint_path) == False: os.mkdir(checkpoint_path)
checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.tar')
if os.path.exists(checkpoint_file) and NEW_MODEL_FLAG == False:
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('Checkpoint File Loaded')




def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        x, y = data
        x, y = x.to(torch.float32), y.to(torch.float32)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 99:
            print('Epoch: {}, Batch; {}, Loss: {}, lr: {}'.format(epoch+1, batch_idx+1, running_loss/100, scheduler.get_last_lr()[0]))
            running_loss = 0.0

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'epoch': epoch + 1
            }
            torch.save(checkpoint, checkpoint_file)

    scheduler.step()
    



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
    for epoch in range(start_epoch, EPOCH_NUM):
        train(epoch)
        test()
