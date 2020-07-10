import numpy as np
from keras.engine import Layer
from tensorflow.contrib.image import dense_image_warp

from keras.layers import Dense, Input, LeakyReLU, ReLU, Lambda
from keras.layers import Conv2D, Flatten, Concatenate, Activation
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.layers import concatenate, Add, Multiply
from model.attention_layers import AttentionLayer
import keras.backend as K


class BilinearSamplingLayer(Layer):
    def __init__(self, image_size, **kwargs):
        self.image_size = image_size
        super(BilinearSamplingLayer, self).__init__(**kwargs)

    def call(self, tensors):
        original_image, predicted_flow = tensors
        return dense_image_warp(original_image, predicted_flow * self.image_size)

    def compute_output_shape(self, tensor):
        input_shape = tensor[0]
        return None, input_shape[1], input_shape[2], input_shape[3]


def index_to_sin_cos(array: np.ndarray, max_index, loop=True, min_theta=0, max_theta=2 * np.pi):
    n = max_index if loop else max_index - 1
    if n == 0:
        theta = np.zeros(array.shape)
    else:
        theta = np.interp(array, xp=[0, n], fp=[min_theta, max_theta])
    cos_values = np.cos(theta)
    sin_values = np.sin(theta)
    return cos_values, sin_values


def get_modified_decoder_layer(x_d0, x_e, current_attention_strategy, current_image_size, pred_flow=None):
    # Skip connection/Attention Strategies
    # (1) U-Net
    if current_attention_strategy == 'u_net':
        x_d = Concatenate()([x_e, x_d0])
        x_e_rearranged = x_e
    # (2) Cross Attention
    elif current_attention_strategy == 'cr_attn' or current_attention_strategy == 'cr':
        c = AttentionLayer(input_h=x_e, mix_concat="concat", k=2)
        x_d = c(x_d0)
        x_e_rearranged = c.o
    # (Not used) Self Attention
    elif current_attention_strategy == 's_attn':
        c = AttentionLayer(input_h=x_d0, mix_concat="concat", k=2)
        x_d = c(x_d0)
        x_e_rearranged = c.o
    # (4) Flow-based Hard Attention
    elif current_attention_strategy == 'h_attn' or current_attention_strategy == 'h':
        x_e_rearranged = BilinearSamplingLayer(current_image_size)([x_e, pred_flow])
        x_d = Concatenate()([x_e_rearranged, x_d0])
    # (5) Attn U-Net
    elif current_attention_strategy == 'u_attn':
        channels = K.int_shape(x_d0)[3] // 2
        g = Conv2D(channels, kernel_size=1, strides=1, padding='same')(x_d0)
        g = BatchNormalization()(g)
        x = Conv2D(channels, kernel_size=1, strides=1, padding='same')(x_e)
        x = BatchNormalization()(x)

        psi = ReLU()(Add()([g, x]))
        psi = Conv2D(1, kernel_size=1, strides=1, padding='same')(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)

        x_e_rearranged = Multiply()([x_e, psi])

        x_d = Concatenate()([x_e_rearranged, x_d0])

    # (0) Vanilla
    else:
        x_d = x_d0
        x_e_rearranged = None

    return x_e_rearranged, x_d
