#!/usr/bin/env python
#coding:utf-8
#simple command: python evaluate_chainer_imagenet.py -g -1 test.txt
#the format of test.txt is like that of train.txt used in chainer/examples/imagenet
#you can run this evaluation code after saving model by chainer/examples/imagenet (for RGB images)
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
import six

import chainer
from chainer import cuda
#import chainer.links as L
from chainer import optimizers
from chainer import serializers

from PIL import Image
import os
import datetime
import json
import multiprocessing
import random
import threading
import six.moves.cPickle as pickle
from six.moves import queue
from chainer import computational_graph
import alex_model_for_classification
import compute_mean, load_image_from_list, util

parser = argparse.ArgumentParser()
parser.add_argument('test', help='Path to test image file or list')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--initmodel', '-init', default='model',
                    help='Initialize the model from given file')
parser.add_argument('--mean', '-m', default='resource/model/mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
group_w = parser.add_mutually_exclusive_group()
group_w.add_argument('--write_f', '-w', action='store_true')
group_w.add_argument('--no-write_f', '-n-w', action='store_false')
parser.set_defaults(write_f=False)
group_p = parser.add_mutually_exclusive_group()
group_p.add_argument('--print_f', '-p', action='store_true')
group_p.add_argument('--no-print_f', '-n-p', action='store_false')
parser.set_defaults(print_f=False)

args = parser.parse_args()

xp = cuda.cupy if args.gpu >= 0 else np

# Prepare model

model = alex_model_for_classification.Alex()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
else:
	print('cannot evaluate model!!')


if args.write_f:
    p = args.initmodel.rsplit('/',1)
    name = '%s/log_%s.txt'%(p[0],p[1])
    print('Write log > %s'%name)
    import csv
    f = open(name,'wb')
    writer = csv.writer(f,delimiter=' ')
    writer.writerow(('path','estimate','correct'))

if not args.print_f:
    print('Don\'t print log')


mean_image = pickle.load(open(args.mean, 'rb'))
test_batchsize = 1 #一度に判定する画像枚数

data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

cropwidth = 256 - model.insize

def load_image_list(path, root):
    images = []
    labels = []
    for line in open(path):
        pair = line.strip().split()
        images.append(os.path.join(root, pair[0]))
        label = util.cls_label(np.float(pair[1]))
        labels.append(np.int32(label))
    return images, labels

def read_image(path, center=False, flip=False):
  # for simple RGB image input
    # Data loading routine
    image = np.asarray(Image.open(path).resize((256,256))).transpose(2, 0, 1)
    top = left = cropwidth / 2
    bottom = model.insize + top
    right = model.insize + left

    image = image[:, top:bottom, left:right].astype(np.float32)
    image -= mean_image[:, top:bottom, left:right]
    image /= 255
    return image

#画像かリストファイルか判定：画像ならエラーが出る
try:
    test_data, test_label = load_image_from_list.for_classification(args.test, args.root)
    orig = True
except:
    test_data, test_label = [args.test], [0]
    orig = False
    
print('test size:',len(test_data))
begin_at = time.time()

def feed_data():
    # Data feeder
    test_x_batch = np.ndarray((test_batchsize, 3, model.insize, model.insize), dtype=np.float32)
    test_y_batch = np.ndarray((test_batchsize,), dtype=np.int32)

    pool = multiprocessing.Pool(args.loaderjob)
    data_q.put("test")
    j = 0
    for path,label in zip(test_data,test_label):
        test_batch_pool = pool.apply_async(read_image,(path,True,False))
        test_y_batch[0] = label
        test_x_batch[0] = test_batch_pool.get()
        data_q.put([path, test_x_batch.copy(), test_y_batch.copy()])
	
    pool.close()
    pool.join()
    data_q.put("end")
def log_result():
    # Logger
    count = 0
    begin_at = time.time()
    while True:
        inp = res_q.get()
        if inp == 'end':
            print(file=sys.stderr)
            break
        elif inp == 'test':
            print(file=sys.stderr)
            train = False
            continue

        path, result, label = inp
        count += 1
        
        print(result)
        print(result[0][0])
        if result[0][0] > result[0][1]:    
            res = 0
        else:
            res = 1
        print(result, res)
        
        if args.print_f:
            if orig:
                print('"%s": %f (orig: %f)'%(path,res,int(label)))
            else:
                print('"%s": %f'%(path,res))
            
        if args.write_f:
            duration = time.time() - begin_at
            throughput = count / duration
            sys.stderr.write(
                '\r{} samples, time: {} ({} images/sec)'
                .format(count, datetime.timedelta(seconds=duration), throughput))
            
            writer.writerow((path,res,label))
            

def test_net():
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'test':  # restart training
            res_q.put('test')
            model.train = False
            continue

        path, data, label = inp
        volatile = 'off' if model.train else 'on'
        x = chainer.Variable(xp.asarray(data), volatile=volatile)
        t = chainer.Variable(xp.asarray(label), volatile=volatile)

        model(x,t)
        res_q.put([path,model.soft.data,int(label)])
        del x, t

# Invoke threads

feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

test_net()
feeder.join()
logger.join()

duration = time.time() - begin_at
print('total time:',duration)

