# coding: utf-8

from __future__ import unicode_literals

from keras import backend as K
# pylint: disable=protected-access
from keras.layers.pooling import _Pooling2D


class GeM2D(_Pooling2D):
    def __init__(self, p, *args, **kwargs):
        self.p = K.constant(p)
        super(GeM2D, self).__init__(*args, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pow(K.pool2d(K.pow(inputs, self.p), pool_size, strides,
                                padding, data_format, pool_mode='avg'), 1 / self.p)
        return output
