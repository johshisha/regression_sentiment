#coding:utf-8

import csv, os, sys
import numpy as np

f = open('../resource/regression_label.txt','rb')
reader = csv.reader(f, delimiter=' ')
reader = list(reader)
f.close()

pos = 0
neg = 0
count = 0

pos_data = []
neg_data = []
for path, label in reader:
    label = float(label)
    if label > 0:
        pos += 1
        pos_data.append((path,label))
    elif label < 0:
        neg += 1
        neg_data.append((path,label))
        
    count += 1
    
print 'all:', count, ', pos:', pos, ', neg:', neg

print 'ratio:: pos:', pos/float(count), ', neg:', neg/float(count)


import random

while(True):
    random.shuffle(pos_data)

    num = neg
    pos_datas = pos_data[:num]
    neg_datas = neg_data[:num]

    print('pos平均：%f'%(np.mean(np.array(map(lambda x: float(x[1]), pos_datas)))))
    print('pos分散：%f'%(np.var(np.array(map(lambda x: float(x[1]), pos_datas)))))
    print('neg平均：%f'%(abs(np.mean(np.array(map(lambda x: float(x[1]), neg_datas))))))
    print('neg分散：%f'%(np.var(np.array(map(lambda x: float(x[1]), neg_datas)))))
    
    print '\ninput ok or no'
    answer = 'ok'#raw_input()
    if answer == 'ok':
        break

f = open('resource/same_number_label.txt','wb')
writer = csv.writer(f, delimiter=' ')
for data in [pos_datas, neg_datas]:
    for row in data:
        writer.writerow(row)
        
f.close()

