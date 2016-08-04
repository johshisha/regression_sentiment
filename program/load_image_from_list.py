#coding: utf-8

import os
import numpy as np
import util

def for_classification(path, root):
    images = []
    labels = []
    for line in open(path):
        pair = line.strip().split()
        images.append(os.path.join(root, pair[0]))
        #このままだと0.5をpositiveとして扱っている
        label = util.cls_label(np.float32(pair[1]))
        labels.append(np.int32(label))
    return images, labels
    
    
def for_regression(path, root):
    images = []
    labels = []
    for line in open(path):
        pair = line.strip().split()
        images.append(os.path.join(root, pair[0]))
        labels.append(np.float32(pair[1]))
    return images, labels
    

#make_cv_lists用
def load_image_list(path):
    images = []
    labels = []
    for line in open(path):
        pair = line.strip().split()
        images.append(pair[0])
        labels.append(np.float32(pair[1]))
    return images, labels
