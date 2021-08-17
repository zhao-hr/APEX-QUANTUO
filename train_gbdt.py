import sklearn
from datasets import MyDataSet
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import joblib
import os

LABEL = 'total_sales'
DATA_PATH = 'data/final_table.csv'
CHECKPOINT_PATH = 'stats/gbdt'
NEW_MODEL_FLAG = True   ### Whether to train a new model instead of loading checkpoint.pkl

model = GradientBoostingClassifier(learning_rate=0.1, subsample=0.9, n_estimators=100, max_depth=5, min_samples_leaf=2)
train_dataset = MyDataSet(DATA_PATH, LABEL, 1)
test_dataset = MyDataSet(DATA_PATH, LABEL, 0)
X_train, y_train = train_dataset.get_data()
X_test, y_test = test_dataset.get_data()
print(X_train.shape, X_test.shape)
len_train = X_train.shape[0]
len_test = X_test.shape[0]

checkpoint_path = os.path.join(CHECKPOINT_PATH, LABEL)
if os.path.exists(checkpoint_path) == False: os.mkdir(checkpoint_path)
checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.pkl')
if os.path.exists(checkpoint_file) and NEW_MODEL_FLAG == False: model = joblib.load(checkpoint_file)


def train():
    print('Training...')
    if NEW_MODEL_FLAG == True: model.fit(X_train, y_train)
    joblib.dump(model, checkpoint_file)


def test():
    predicted = model.predict(X_test)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len_test):
        #print(y_test[i], predicted[i])
        if y_test[i] == 1:
            if predicted[i] == 1: TP += 1
            else: FN += 1
        else:
            if predicted[i] == 1: FP += 1
            else: TN += 1
    TPRate = TP / (TP + FN)
    FPRate = FP / (FP + TN)
    AUC = metrics.roc_auc_score(y_test, predicted)
    print('TPRate: ', TPRate)
    print('FPRate: ', FPRate)
    print('AUC: ', AUC)
    

if __name__ == '__main__':
    train()
    test()
