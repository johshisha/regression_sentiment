#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm
from sklearn import cross_validation
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler
import argparse, os, pickle
import util


parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('feature_file_dir', help='Path to data image-label list file directory')
parser.add_argument('test_k', type=int, help='Number of k fold (1-5)')
parser.add_argument('--num', '-n', type=int, default=128, help='Number of dimention for pca')
group_t = parser.add_mutually_exclusive_group()
group_t.add_argument('--train_f', '-t', action='store_true')
group_t.add_argument('--no-train_f', '-n-t', action='store_false')
parser.set_defaults(train_f=False)
args = parser.parse_args()


k = 5
test_k = args.test_k

out_path = os.path.join(args.feature_file_dir, 'fc7_cv_pca%d'%test_k)
out_model = os.path.join(args.feature_file_dir, 'pca%d.pkl'%test_k)
out_scale = os.path.join(args.feature_file_dir, 'scale%d.pkl'%test_k)

data_train = []
label_train = []
path_train = []
for i in range(1, k+1):
    if i != test_k:
        path = args.feature_file_dir + '/fc7_cv_list%d.txt'%i
        data_list, label_list, path_list = util.load_fc7_features_r(path)
        data_train.extend(data_list)
        label_train.extend(label_list)
        path_train.extend(path_list)

path = args.feature_file_dir + '/fc7_cv_list%d.txt'%test_k
data_test, label_test, path_test = util.load_fc7_features_r(path)

#data_train, data_test, label_train, label_test = \
#    cross_validation.train_test_split(data_test, label_test, test_size=0.4)
    
print('train size', len(data_train))

print('train', np.array(data_train).shape)
print('test', np.array(data_test).shape)


if os.path.isfile(out_model) and not args.train_f:
    # 予測モデルを復元
    pca = pickle.load(open(out_model, 'rb'))
else:
    # 主成分分析による次元削減
    pca = decomposition.PCA(n_components = args.num)
    pca.fit(data_train)

pca_train = pca.transform(data_train)
pca_test = pca.transform(data_test)
del data_train, data_test

if not (os.path.isfile(out_model) and not args.train_f):
    #モデルをシリアライズ
    f = open(out_model, 'wb')
    pickle.dump(pca, f)
    f.close()
    
del pca

print('train',pca_train.shape)
print('test',pca_test.shape)

if os.path.isfile(out_scale) and not args.train_f:
    # 予測モデルを復元
    scaler = pickle.load(open(out_scale, 'rb'))
else:
    scaler = MinMaxScaler()
    scaler.fit(pca_train)
    
scale_train = scaler.transform(pca_train)
scale_test = scaler.transform(pca_test)
del pca_train, pca_test

if not (os.path.isfile(out_scale) and not args.train_f):
    #モデルをシリアライズ
    f = open(out_scale, 'wb')
    pickle.dump(scaler, f)
    f.close()
    
del scaler

f = open(out_path + '_train.txt','wb')
writer = csv.writer(f,delimiter=' ')
writer.writerow(('path','label','features'))
for path, label, data in zip(path_train, label_train, scale_train):
    data = list(data)
    data.insert(0, float(label))
    data.insert(0, path)
    writer.writerow(data)
f.close()

f = open(out_path + '_test.txt','wb')
writer = csv.writer(f,delimiter=' ')
writer.writerow(('path','label','features'))
for path, label, data in zip(path_test, label_test, scale_test):
    data = list(data)
    data.insert(0, float(label))
    data.insert(0, path)
    writer.writerow(data)
f.close()

