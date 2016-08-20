import numpy

from chainer import function
from chainer import cuda
from chainer.utils import type_check
import math
from convex_function import convex

cp = cuda.cupy

class MeanSquaredError(function.Function):

    def __init__(self):
        self.a=3
        self.b=2
     
     
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
        diff = self.diff.ravel() * map(lambda p: convex(p[0]), x0)
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        #for i in range(len(x0)):
        #    x0[i] = x0[i] * convex(x0[i][0], a=6, b=3)
        #    x1[i] = x1[i] * convex(x1[i][0], a=6, b=3)
        con = map(lambda x: [convex(x[0], a=self.a, b=self.b)], x0)
        for i in range(len(x0)):
            x0[i][0] = (x0[i][0] - x1[i][0]) * con[i][0]     
        self.diff = x0
        #self.diff = (x0 - x1) * con
        diff = self.diff.ravel()
        #n = map(lambda p: convex(p[0]), x0)
        #diff = numpy.array(map(lambda x: diff[x] * n[x], range(len(n))))
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, inputs, gy):
        gys = gy[0] * convex(gy[0],  a=self.a, b=self.b)
        
        coeff = gys * gy[0].dtype.type(2. / self.diff.size)
        gx0 = coeff * self.diff
        return gx0, -gx0


def mean_squared_error(x0, x1):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1)
    

