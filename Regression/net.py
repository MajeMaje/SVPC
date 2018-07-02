import six

import chainer
import chainer.functions as F
import chainer.links as L


import numpy as np

class MLP(chainer.Chain):
    def __init__(self, n_in):
        super(MLP, self).__init__()
        with self.init_scope():
            # liner
            self.l1 = L.Linear(n_in, 192)
            self.l2 = L.Linear(192, 64)
            self.l3 = L.Linear(64, 32)
            self.l4 = L.Linear(32, 1)
            # bn 
            self.bn1 = L.BatchNormalization(192)
            self.bn2 = L.BatchNormalization(64)
            self.bn3 = L.BatchNormalization(32)

    def __call__(self, x ,y):
        x = F.dropout(x,ratio=0.1)
        h1 = F.relu(self.bn1(self.l1(x)))
        h1 = F.dropout(h1)
        h1 = F.relu(self.bn2(self.l2(h1)))
        h1 = F.dropout(h1)
        h1 = F.relu(self.bn3(self.l3(h1)))
        h1 = F.dropout(h1)
        y_pred = F.relu(self.l4(h1))
        
        self.loss = F.sqrt(F.mean_squared_error(y.reshape(y_pred.shape), y_pred))

        return self.loss
