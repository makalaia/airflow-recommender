from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

import numpy as np
import pickle

from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

with open('data/movielens/ml-100k.pickle', mode='rb') as file:
    movielens = pickle.load(file)

train = movielens['train']
test = movielens['test']

permutations = np.random.permutation(train.nnz)
for i in range(1, 11):
    print('EPOCH: %s' % i)
    p = permutations[:int(train.nnz/10*i)]
    rows, cols, dta = train.row[p], train.col[p], train.data[p]
    train_set = coo_matrix((dta, (rows, cols)), shape=train.shape)

    item_features = movielens['item_features']
    item_feature_labels = movielens['item_feature_labels']
    item_labels = movielens['item_labels']

    model = LightFM(learning_rate=0.05, loss='bpr')
    model.fit(train_set, epochs=10)

    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision = precision_at_k(model, test, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
    print('--------------------------------------------', end='\n\n')
