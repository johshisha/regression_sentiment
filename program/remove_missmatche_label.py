#coding: utf-8
import pickle, sys, argparse, csv, shutil, os
from collections import defaultdict
from PIL import Image
import numpy as np

def dd():
    return defaultdict(int)
  
def regression_label_d(d, file):
    f = open(file, 'rb')
    reader = csv.reader(f, delimiter=' ')

    split_file = file.rsplit('/', 1)
    w_file = os.path.join(split_file[0], 'removed_%s'%split_file[1])
    writer = csv.writer(open(w_file,'wb'), delimiter=' ')

    for k, v in reader:
        labels = d[k].keys()
        if is_not_missmatche_image(labels):   
            writer.writerow([k,v])
        
def is_not_missmatche_image(labels):
    pos = False
    neg = False
    for l in labels:
        #judge positive
        if 'ositive' in l:
            pos = True
        #judge negative
        elif 'egative' in l:
            neg = True
    if pos and neg:
        return False
    else:
        return True
        
        
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-p','--pkl',required=False, default='sentiment_degree.pkl',\
                     help='path to the pickle file')
    ap.add_argument('-f','--file',required=False, default='.',\
                     help='path to the target file')
    args = vars(ap.parse_args())
    
    d = pickle.load(open(args['pkl'],'rb'))

    """
    while(True):
        print 'input order'
        order = raw_input()
        if order == 'exit':
            break
        exec(order)
    """
    regression_label_d(d, args['file'])

if __name__ == '__main__':
    main(sys.argv)
  
