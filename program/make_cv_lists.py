#coding:utf-8
import argparse, csv
import numpy as np
import load_image_from_list

parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('image_file', help='Path to data image-label list file')
parser.add_argument('--k_fold', '-k', default=5, type=int,
                    help='Number of k for k fold validation')
args = parser.parse_args()



# Prepare dataset
data_list, label_list = load_image_from_list.load_image_list(args.image_file)

k = args.k_fold

#data_list, label_list = remove_broken_image(data_list, label_list, args.root)

all_len = len(data_list)
fold_len = all_len / k
print(all_len, fold_len)


datas = zip(data_list, label_list)
import random
random.shuffle(datas)

data_lists = []
label_lists = []

for i in range(k):
    f = fold_len * i
    l = fold_len * (i+1)
    print(f, l)
    d = datas[f:l]
    data_lists.append(map(lambda x: x[0], d))
    label_lists.append(map(lambda x: x[1], d))
    print(len(data_lists))


for i in range(k):
    data = data_lists[i]
    label = label_lists[i]
    tr_f = open('resource/cv_lists/cv_list%d.txt'%(i+1) ,'wb')
    tr_li = csv.writer(tr_f, delimiter=' ')
    print(len(data), len(label))
    count = 0
    for line in zip(data,label):
        tr_li.writerow(line)
        count += 1
    print(count)
    tr_f.close()

