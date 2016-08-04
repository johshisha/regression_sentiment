#coding: utf-8

import csv, os, sys

def load_fc7_features_c(filepath):
    f = open(filepath, 'rb')
    reader = csv.reader(f ,delimiter=' ')
    header = next(reader)
    feat = []
    label = []
    path = []
    for row in reader:
        p = row[0]
        l = cls_label(float(row[1]))
        label.append(int(l))
        feat.append(map(float, row[2:]))       
        path.append(p)
    return feat, label, path
    
def load_fc7_features_r(filepath):
    f = open(filepath, 'rb')
    reader = csv.reader(f ,delimiter=' ')
    header = next(reader)
    feat = []
    label = []
    path = []
    for row in reader:
        p = row[0]
        label.append(float(row[1]))
        feat.append(map(float, row[2:]))       
        path.append(p)
    return feat, label, path
    
def cls_label(score):
    if score < 0.0:
        label = -1
    else:
        label = 1
    return label
