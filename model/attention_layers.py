from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from keras.constraints import MinMaxNorm

'''
This code is slightly modified SAGAN code from
https://github.com/kiyohiro8/SelfAttentionGAN/blob/master/SelfAttentionLayer.py
'''


class AttentionLayer(Layer):
    def __init__(self, input_h, mix_concat='concat', k=8, **kwargs):
        '''
        This is a
        :param input_h: Input hidden layer
        :param mix_concat: how to append attention applied layer. Only concat is used.
        :param k: parameter for channel size. Smaller k means more channels.
        :param kwargs:
        '''
        self.input_h = input_h
        self.mix_concat = mix_concat
        self.k = k
        self.beta = None
        self.o = None
        self.gamma = None
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        channels = input_shape[3]
        channels2 = K.int_shape(self.input_h)[3]
        self.channels = channels
        self.w = input_shape[1]
        self.f = self.add_weight(name='f', shape=(1, 1, channels2, channels // self.k), initializer='uniform', trainable=True)
        self.g = self.add_weight(name='g', shape=(1, 1, channels, channels // self.k), initializer='uniform', trainable=True)
        if self.mix_concat == 'mix':
            self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='uniform', trainable=True)
        elif self.mix_concat == 'weighted_mix':
            self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='uniform', trainable=True, constraint=MinMaxNorm(min_value=0.0, max_value=1.0))
        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        def hw_flatten(x):
            return tf.reshape(x, shape=[K.shape(x)[0], -1, K.shape(x)[-1]])

        channels = K.int_shape(self.input_h)[3]
        w = self.w
        f = K.conv2d(self.input_h, self.f, strides=(1, 1), padding="same")
        g = K.conv2d(x, self.g, strides=(1, 1), padding="same")

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
        beta = tf.nn.softmax(s)
        self.beta = beta

        o = tf.matmul(beta, hw_flatten(self.input_h))
        o = K.reshape(o, (K.shape(x)[0], w, w, channels))
        self.o = o

        # only concat is used.
        if self.mix_concat == 'concat':
            x = K.concatenate([o, x])
        elif self.mix_concat == 'mix':
            x = self.gamma * o + x
        elif self.mix_concat == 'weighted_mix':
            x = self.gamma * o + (1 - self.gamma) * x
        return x

    def compute_output_shape(self, input_shape):
        if self.mix_concat == 'concat':
            return None, self.w, self.w, int(self.channels + K.int_shape(self.input_h)[3])
        else:
            return input_shape