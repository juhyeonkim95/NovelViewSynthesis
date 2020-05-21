import tensorflow as tf
import numpy as np
from keras.layers import Layer
from tensorflow.contrib.image import dense_image_warp


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