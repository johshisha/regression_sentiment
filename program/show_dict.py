#coding: utf-8
import pickle, sys, argparse
from collections import defaultdict

def dd():
    return defaultdict(int)
  
def show_d(d,num=-1):
    count = 0
    for k,v in d.items():
        if count == num:
            break
        print k,v.items()
        count += 1
        
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-n','--num',required=True,help='show data number')
    ap.add_argument('-f','--file',required=True,help='path to the pickle file')
    args = vars(ap.parse_args())
    
    d = pickle.load(open(args['file'],'rb'))
    show_d(d, int(args['num']))

if __name__ == '__main__':
    main(sys.argv)
  
