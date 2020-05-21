from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
#from utils import *
from keras.constraints import MinMaxNorm


class SelfAttentionLayer(Layer):
    def __init__(self, k=8, m = 2, **kwargs):
        self.k = k
        self.m = m
        self.beta = None
        self.input_h = None
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        channels = input_shape[3]
        self.channels = channels
        self.w = input_shape[1]
        self.f = self.add_weight(name='f', shape=(1, 1, channels, channels // self.k), initializer='uniform', trainable=True)
        self.g = self.add_weight(name='g', shape=(1, 1, channels, channels // self.k), initializer='uniform', trainable=True)
        self.h = self.add_weight(name='h', shape=(1, 1, channels, channels // self.m), initializer='uniform', trainable=True)
        self.o = self.add_weight(name='o', shape=(1, 1, channels // self.m, channels), initializer='uniform', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1, ), initializer='uniform', trainable=True)
        super(SelfAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        def hw_flatten(x):
            return tf.reshape(x, shape=[K.shape(x)[0], -1, K.shape(x)[-1]])
        self.input_h = x
        channels = self.channels
        w = self.w

        f = K.conv2d(x, self.f, strides=(1,1), padding="same")
        f = K.pool2d(f, pool_size=(2, 2), strides=(2, 2), padding="same")

        g = K.conv2d(x, self.g, strides=(1, 1), padding="same")

        h = K.conv2d(x, self.h, strides=(1, 1), padding="same")
        h = K.pool2d(h, pool_size=(2, 2), strides=(2, 2), padding="same")

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
        beta = tf.nn.softmax(s)
        self.beta = beta
        o = tf.matmul(beta, hw_flatten(h))
        o = K.reshape(o, (K.shape(x)[0], w, w, channels // 2))
        o = K.conv2d(o, self.o, strides=(1, 1), padding="same")

        x = self.gamma * o + x
        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return input_shape


class AttentionLayer(Layer):
    def __init__(self, input_h, mix_concat='mix', k=8, u_value=None, **kwargs):
        self.input_h = input_h
        print(K.shape(input_h), type(input_h))
        self.mix_concat = mix_concat
        self.k = k
        self.beta = None
        self.o = None
        self.u_value = u_value
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        channels = input_shape[3]
        channels2 = K.int_shape(self.input_h)[3]
        self.channels = channels
        self.w = input_shape[1]

        if self.u_value is None:
            self.f = self.add_weight(name='f', shape=(1, 1, channels2, channels // self.k),
                                     initializer='uniform', trainable=True)
            self.g = self.add_weight(name='g', shape=(1, 1, channels, channels // self.k),
                                     initializer='uniform', trainable=True)
        else:
            from keras.initializers import RandomUniform
            a = self.u_value
            self.f = self.add_weight(name='f', shape=(1, 1, channels2, channels // self.k), initializer=RandomUniform(minval=-a, maxval=a, seed=None), trainable=True)
            self.g = self.add_weight(name='g', shape=(1, 1, channels, channels // self.k), initializer=RandomUniform(minval=-a, maxval=a, seed=None), trainable=True)

        if self.mix_concat == 'mix':
            self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='uniform', trainable=True)
        elif self.mix_concat == 'weighted_mix':
            self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='uniform', trainable=True,
                                         constraint=MinMaxNorm(min_value=0.0, max_value=1.0))

        elif self.mix_concat == 'planar_gamma':
            self.gamma = self.add_weight(name='gamma', shape=(self.w,self.w), initializer='uniform', trainable=True)

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

