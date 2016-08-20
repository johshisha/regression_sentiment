#!/usr/bin/env python
import argparse
import os
import sys

import numpy
from PIL import Image
import six.moves.cPickle as pickle


def compute_mean(dataset, path='resource/model/mean.npy', force = False):
    if os.path.isfile(path) and not force:
        mean = pickle.load(open(path, 'rb'))
        return mean

    sum_image = None
    count = 0

    for filepath in dataset:
        try:
            im = Image.open(filepath).resize((256,256))
            image = numpy.asarray(im).transpose(2, 0, 1)
            if sum_image is None:
                sum_image = numpy.ndarray(image.shape, dtype=numpy.float32)
                sum_image[:] = image
            else:
                sum_image += image
            count += 1
            sys.stderr.write('\r{}'.format(count))
            sys.stderr.flush()
        except:
            print filepath
        
    
    sys.stderr.write('\n')

    mean = sum_image / count
    pickle.dump(mean, open(path, 'wb'), -1)
    return mean

import sys, load_image_from_list 
if __name__ == '__main__':
    for test_k in range(5):
        test_k += 1
        root = 'resource/images'
        k = 5
        image_file_dir = 'resource/cv_lists'

        mean_path = image_file_dir + '/removed_remove_%d_mean.npy'%test_k
        print mean_path
        data_train = []
        label_train = []
        for i in range(1, k+1):
            if i != test_k:
                path = image_file_dir + '/removed_cv_list%d.txt'%i
                data_list, label_list = load_image_from_list.for_regression(path, root)
                data_train.extend(data_list)
                label_train.extend(label_list)
                
        mean_image = compute_mean(data_train, path=mean_path)
