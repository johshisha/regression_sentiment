#coding:utf-8

import csv, argparse,os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import util

parser = argparse.ArgumentParser()
parser.add_argument('log_file', help='Path to log file')
parser.add_argument('--graph', dest='graph', action='store_true')
parser.set_defaults(graph=False)
args = parser.parse_args()

corrects = defaultdict(int)
same = defaultdict(int)
for i in range(1,6):
    p = 'resource/cv_lists/cv_list%d.txt'%i
    print p
    reader = csv.reader(open(p ,'rb'), delimiter=' ')
    reader = list(reader)
    
    for path, correct in reader:
        corrects[round(float(correct),1)] += 1
    
    p = 'same_number/resource/cv_lists/cv_list%d.txt'%i
    print p
    reader2 = csv.reader(open(p ,'rb'), delimiter=' ')
    reader2 = list(reader2)

    for path, correct in reader2:
        same[round(float(correct),1)] += 1
    
corrects_l = sorted(corrects.items())
same_l = sorted(same.items())
plt.plot(map(lambda x: x[0], corrects_l), map(lambda x: x[1], corrects_l), label='before')
plt.plot(map(lambda x: x[0], same_l), map(lambda x: x[1], same_l), label='after')

plt.legend(loc='upper left')
plt.xlim([0.0,1.0])
plt.show()


