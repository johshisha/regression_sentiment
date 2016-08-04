#coding:utf-8

import csv, argparse
import numpy as np
import util

parser = argparse.ArgumentParser()
parser.add_argument('log_file', help='Path to log file')
parser.add_argument('--graph', dest='graph', action='store_true')
parser.set_defaults(graph=False)
args = parser.parse_args()

reader = csv.reader(open(args.log_file,'rb'), delimiter=' ')
header = next(reader)
reader = list(reader)

if args.graph:
    diff_l = []
    abs_diff_l = []
    max_d = 0
    for path, estimate, correct in reader:
        diff = float(estimate) - float(correct)
        diff_l.append(diff)
        abs_diff_l.append(np.absolute(diff))
        max_d = max(max_d, np.absolute(diff))
        
    diff_ary = np.array(diff_l)
    abs_diff_ary = np.array(abs_diff_l)

    print('最大誤差：%f'%(max_d))
    print('誤差(絶対値)平均：%f'%(np.mean(abs_diff_ary)))
    print('誤差平均：%f'%(np.mean(diff_ary)))
    print('誤差分散：%f'%(np.var(diff_ary)))
    print('予測平均：%f'%(np.mean(np.array(map(lambda x: float(x[1]), reader)))))
    print('予測分散：%f'%(np.var(np.array(map(lambda x: float(x[1]), reader)))))
    print('ラベル平均：%f'%(np.mean(np.array(map(lambda x: float(x[2]), reader)))))
    print('ラベル分散：%f'%(np.var(np.array(map(lambda x: float(x[2]), reader)))))

    import matplotlib.pyplot as plt
    from collections import defaultdict

    data = defaultdict(int)
    d = []
    estimates = defaultdict(int)
    corrects = defaultdict(int)
    data[0.0]
    estimates[0.0]
    corrects[0.0]
    for path, estimate, correct in reader:
        diff = float(estimate) - float(correct)
        data[round(diff,1)] += 1
        d.append(diff)
        
        estimates[round(float(estimate),1)] += 1
        corrects[round(float(correct),1)] += 1
        
    data_l = sorted(data.items())
    estimates_l = sorted(estimates.items())
    corrects_l = sorted(corrects.items())
    #plt.plot(map(lambda x: x[0], data_l), map(lambda x: x[1], data_l))
    plt.plot(map(lambda x: x[0], estimates_l), map(lambda x: x[1], estimates_l), label='estimate')
    plt.plot(map(lambda x: x[0], corrects_l), map(lambda x: x[1], corrects_l), label='correct')
    plt.hist(d, bins=20, range=(-1,1), label='error hist')

    plt.legend(loc='upper left')
    plt.xlim([-1.0,1.0])
    plt.show()


perform = {(1,1):0, (1,-1):0, (-1,1):0, (-1,-1):0}

label = {(1,1):'(正)予測：P、正解：P', (1,-1):'(誤)予測：P、正解：N',(-1,1):'(誤)予測：N、正解：P',(-1,-1):'(正)予測：N、正解：N'} 
for path, estimate, correct in reader:
    est_label = util.cls_label(float(estimate))
    cor_label = util.cls_label(float(correct))

    key = (est_label, cor_label)
    perform[key] += 1

for x in perform.keys():
    print('%s: %d\t\t'%(label[x], perform[x]))


print('精度：%f'%( (perform[(-1,-1)]+perform[(1,1)]) / float(sum(perform.values())) ))
