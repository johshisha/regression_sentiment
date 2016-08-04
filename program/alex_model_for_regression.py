import chainer
import chainer.functions as F
import chainer.links as L


class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1),
            #bn1 = L.BatchNormalization(96),
            #bn2 = L.BatchNormalization(256),
            #bn3 = L.BatchNormalization(384),
            #bn4 = L.BatchNormalization(384),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h1 = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h2 = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h1))), 3, stride=2)
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.max_pooling_2d(F.relu(self.conv5(h4)), 3, stride=2)
        h6 = F.dropout(F.relu(self.fc6(h5)), train=self.train)
        h7 = F.dropout(F.relu(self.fc7(h6)), train=self.train)
        h8 = self.fc8((h7))
        self.h = h8
        self.loss = F.mean_squared_error(h8, t)
        
        #self.accuracy = F.accuracy(h, t)
        return self.loss
    

