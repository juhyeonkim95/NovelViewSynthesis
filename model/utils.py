import numpy as np
from keras.engine import Layer
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


def index_to_sin_cos(array: np.ndarray, max_index, loop=True, min_theta=0, max_theta=2 * np.pi):
    n = max_index if loop else max_index - 1
    if n == 0:
        theta = np.zeros(array.shape)
    else:
        theta = np.interp(array, xp=[0, n], fp=[min_theta, max_theta])
    cos_values = np.cos(theta)
    sin_values = np.sin(theta)
    return cos_values, sin_values