#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import argparse, os, pickle
import util


parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('feature_file_dir', help='Path to data image-label list file directory')
parser.add_argument('test_k', type=int, help='Number of k fold (1-5)')
group_w = parser.add_mutually_exclusive_group()
group_w.add_argument('--write_f', '-w', action='store_true')
group_w.add_argument('--no-write_f', '-n-w', action='store_false')
parser.set_defaults(write_f=False)
group_t = parser.add_mutually_exclusive_group()
group_t.add_argument('--train_f', '-t', action='store_true')
group_t.add_argument('--no-train_f', '-n-t', action='store_false')
parser.set_defaults(train_f=False)
args = parser.parse_args()
    

k = 5
test_k = args.test_k

out_path = 'resource/cv_svr_pca_model/remove_%d'%test_k
if not os.path.isdir(out_path):
    os.mkdir(out_path)


#out_model = args.out
#out_state = args.outstate
out_model = os.path.join(out_path, 'model')
out_params = os.path.join(out_path, 'params')

if args.write_f:
    p = out_model.rsplit('/',1)
    name = '%s/result_%s.txt'%(p[0],p[1])
    print('Write result > %s'%name)
    import csv
    f = open(name,'wb')
    writer = csv.writer(f,delimiter=' ')
    writer.writerow(('path','estimate','correct'))


data_train = []
label_train = []
path_train = []

path = args.feature_file_dir + '/fc7_cv_pca%d_train.txt'%test_k
data_list, label_list, path_list = util.load_fc7_features_r(path)
data_train.extend(data_list)
label_train.extend(label_list)
path_train.extend(path_list)
        
del path_train

path = args.feature_file_dir + '/fc7_cv_pca%d_test.txt'%test_k
data_test, label_test, path_test = util.load_fc7_features_r(path)
 
print('train size', len(data_train))


if os.path.isfile(out_model) and not args.train_f:
    # 予測モデルを復元
    print 'load trained model: %s'%out_model
    reg_max = joblib.load(out_model)
else:
    #RBFカーネルのパラメーターγと罰則Cを複数個作ってその中で(スコアの意味で）良い物を探索(カーネルもパラメーターとして使用可能)
    print 'doing grid search ...'
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10**i for i in range(-4,0)], 'C': [10**i for i in range(1,4)]}]
    gscv = GridSearchCV(svm.SVR(), tuned_parameters, n_jobs=8, cv=5, scoring="mean_squared_error")

    #gscv = svm.SVR(kernel='rbf', C=1)
    gscv.fit(data_train, label_train)
    
    #一番スコア悪い&良い奴を出す
    reg_max = gscv.best_estimator_
    print reg_max.get_params
    #全トレーニングデータを使って再推計
    reg_max.fit(data_train, label_train)


del data_train, label_train

result = reg_max.predict(data_test)

if args.write_f:
    for p,r,l in zip(path_test,result,label_test):
        writer.writerow((p,r,l))

# 予測モデルをシリアライズ
joblib.dump(reg_max, out_model)
f = open(out_params, 'wb')
pickle.dump(reg_max.get_params(), f)
f.close()

