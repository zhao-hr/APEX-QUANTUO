import pandas as pd
import numpy as np
import copy


DATA_PATH = 'data/final_table.csv'
POS_SAMPLE = 0.5  ### Assign the biggest 50% elements as 1 and the others as 0
TRAIN_PROP = 0.7  ### Apply 70% of the data for training and the others for testing


### Find the N-th maximum element in ndarray 'list' ###
def NMax(list, N):
    list_copy = copy.deepcopy(list)
    length = list_copy.shape[0]
    list_copy.sort()
    return list_copy[length - N]

### Assign the biggest N-th elements in ndarray 'list' as 1 and the others as 0 ###
def NMaxPos(list, N):
    nmax = NMax(list, N)
    if nmax == 0:
        list[list <= 0] = 0
        list[list > 0] = 1
    else:
        list[list < nmax] = 0
        list[list >= nmax] = 1

### Count the number of elements in ndarray 'list' which is not 'nan'(represented as -1) ###
def CntNotNan(list):
    return list[list != -1].shape[0]



class MyDataSet(object):
    def __init__(self, data_path) -> None:
        super().__init__()

        '''
        'self.length' is the number of rows of the table
        'self.width' is the number of columns of the table
        'self.header' is the header of the table with a shape of [1, self.width]
        'self.data' is the data in the table with a shape of [self.length, self.width]

        '__len__(self)' returns the number of rows of the table
        'getWidth(self)' returns the number of columns of the table
        '__getitem__(self, index)' returns the index-th row including its features of shape [1, 18] and its label
        'getColumn(self, index)' returns the index-th column
        'train_data(self)' returns the training data including features of shape [train_length, 18] and labels of shape [train_length, 1]
        'test_data(self)' returns the testing data including features of shape [test_length, 18] and labels of shape [test_length, 1]
            where train_length = self.length * TRAIN_PROP, test_length = self.length - train_length
        '''

        fd = pd.read_csv(data_path)
        self.header = fd.columns
        self.data = np.array(fd)
        self.data[np.isnan(self.data)] = -1
        self.length = self.data.shape[0]
        self.width = self.data.shape[1]

        #for i in range(self.width):
        #    print(self.header[i], ': ', CntNotNan(self.data[:,i]))
        
        #for i in range(1, self.width):
        #    list = self.data[:,i]
        #    if i == 1: list[list != 1] == 0
        #    else: NMaxPos(list, int(CntNotNan(list) * POS_SAMPLE))

        lst = self.data[:,9]
        NMaxPos(lst, int(CntNotNan(lst) * POS_SAMPLE))

    def __len__(self):
        return self.length
    
    def getWidth(self):
        return self.width
    
    def __getitem__(self, index):
        list = copy.deepcopy(self.data[index])
        label = list[9]
        return np.delete(list, [0,9]), label
    
    def getColumn(self, index):
        list = copy.deepcopy(self.data[:,index])
        return list
    
    def train_data(self):
        data = copy.deepcopy(self.data)
        data = np.delete(data, [0,9], axis=1)
        length = int(self.length * TRAIN_PROP)
        return data[0:length], self.data[0:length, 9]
    
    def test_data(self):
        data = copy.deepcopy(self.data)
        data = np.delete(data, [0,9], axis=1)
        length = int(self.length * TRAIN_PROP)
        return data[length:self.length], self.data[length:self.length, 9]




if __name__ == '__main__':
    dataset = MyDataSet(DATA_PATH)
    train_X, train_y = dataset.train_data()
    test_X, test_y = dataset.test_data()
    total_tax = dataset.getColumn(9)
    item31, item31_label = dataset[31]

    print(len(dataset), dataset.getWidth())
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)
    print(total_tax.shape)
    print(item31.shape, item31_label)
