import numpy

from chainer import function
from chainer.utils import type_check
import math


class MeanSquaredError(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = (x0 - x1)
        diff = self.diff.ravel()
        n = map(lambda p: divided_case(p[0][0],p[1][0]),zip(x0,x1))
        diff = numpy.array(map(lambda x: diff[x] * n[x], range(len(n))))
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        for i in range(len(x0)):
            x0[i] = x0[i] * divided_case(x0[i][0], x1[i][0])
            x1[i] = x1[i] * divided_case(x0[i][0], x1[i][0])
        
        self.diff = (x0 - x1)
        diff = self.diff.ravel()
        #n = map(lambda p: divided_case(p[0][0],p[1][0]),zip(x0,x1))
        #diff = numpy.array(map(lambda x: diff[x] * n[x], range(len(n))))
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.diff.size)
        gx0 = coeff * self.diff
        return gx0, -gx0


def mean_squared_error(x0, x1):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1)
    
from convex_function import convex
def divided_case(y, t):
    #print 'y:',y,', t:',t, 
    y_c = label_c(y)
    t_c = label_c(t)
    #print ',diff:',(y-t)
    
    if y_c != t_c:
        return 2
    else:
        return 1
    
def label_c(a):
    if a >= 0.0:
        return 1
    elif a < 0.0:
        return -1
    
    

