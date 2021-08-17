import sklearn
from datasets import MyDataSet
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import metrics
import joblib
import os

LABEL = 'total_tax'
DATA_PATH = 'data/final_table.csv'
CHECKPOINT_PATH = 'stats/gbdt'
NEW_MODEL_FLAG = True   ### Whether to train a new model instead of loading checkpoint.pkl

model = GradientBoostingRegressor(learning_rate=0.1, subsample=1, n_estimators=100)
train_dataset = MyDataSet(DATA_PATH, LABEL, 1)
test_dataset = MyDataSet(DATA_PATH, LABEL, 0)
X_train, y_train = train_dataset.get_data()
X_test, y_test = test_dataset.get_data()
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

    AUC = metrics.roc_auc_score(y_test, predicted)

    print('AUC: ', AUC)
    

if __name__ == '__main__':
    train()
    test()
