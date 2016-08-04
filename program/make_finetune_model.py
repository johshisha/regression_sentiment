from chainer import link
from chainer.links.caffe import CaffeFunction
from chainer import serializers

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-m','--model',required=True, help='model name')
ap.add_argument('-i','--imp',required=True, help='import module name')
args = vars(ap.parse_args())

model = args['model']
module = args['imp']

exec('from %s import Alex'%module)


def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy %s' % child.name

print 'load Alex caffemodel'
ref = CaffeFunction('resource/bvlc_alexnet.caffemodel')
alex = Alex()
print 'copy weights'
copy_model(ref, alex)

print 'save "alex.model"'
serializers.save_npz('resource/alex_%s.model'%model, alex)
