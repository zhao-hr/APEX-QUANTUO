from operator import mod
from datasets import MyDataSet
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import metrics
import joblib
import os

LABEL = 'total_tax'
DATA_PATH = 'data/final_table.csv'
CHECKPOINT_PATH = 'stats/gbdt'
K = 500

model = GradientBoostingRegressor()
dataset = MyDataSet(DATA_PATH, LABEL, 0)
X_test, y_test = dataset.get_data()
len_test = X_test.shape[0]

checkpoint_path = os.path.join(CHECKPOINT_PATH, LABEL)
if os.path.exists(checkpoint_path) == False: os.mkdir(checkpoint_path)
checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.pkl')
if os.path.exists(checkpoint_file): 
    model = joblib.load(checkpoint_file)
else:
    print('No Trained Model!')


def test():
    predicted = model.predict(X_test)

    for i in range(len_test):
        print(y_test[i], predicted[i])
    AUC = metrics.roc_auc_score(y_test, predicted)
    index = predicted.argsort()
    index = index[::-1]
    cnt = 0
    for i in range(K):
        if y_test[index[i]] == 1: cnt += 1

    print('AUC: ', AUC)
    print('Hit@{}: {}'.format(K, cnt/K))


if __name__ == '__main__':
    test()
