import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from fastFM import als
from sklearn.linear_model import Ridge
import scipy.sparse as sp

user_item_pair = []
rating = []
index = 0

for line in open("train_vec 3.txt"):
    line = line.strip().split(':')
    feature_list = []
    feature_string = line[0].strip().strip('[').strip(']')
    for feature in feature_string.split(','):
        feature = feature.strip()
        feature_list.append(float(feature))
    user_item_pair.append(feature_list)
    rating.append(float(line[1].strip()))
    print(index)
    index += 1
user_item_pair = np.asarray(user_item_pair)
rating = np.asarray(rating)

print('read finished')

# clf = KernelRidge(alpha=1.0, kernel = 'poly', degree = 3)
# clf.fit(user_item_pair, rating)
clf = MLPRegressor(hidden_layer_sizes=(8, 5), learning_rate='invscaling', max_iter=500)

# clf = SVR(kernel='poly',degree=2, max_iter=3000)
# user_item_pair = sp.csc_matrix(user_item_pair)
# clf = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
# clf = Ridge(solver='sag', alpha=1.0)
clf.fit(user_item_pair, rating)


joblib.dump(clf, 'DL_model_nor.pkl')



