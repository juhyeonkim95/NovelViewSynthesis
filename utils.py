import numpy as np

def index_to_sin_cos(array: np.ndarray, max_index, loop=True, min_theta=0, max_theta=2 * np.pi):
    n = max_index if loop else max_index - 1
    if n == 0:
        theta = np.zeros(array.shape)
    else:
        theta = np.interp(array, xp=[0, n], fp=[min_theta, max_theta])
    cos_values = np.cos(theta)
    sin_values = np.sin(theta)
    return cos_values, sin_values


def hw_flatten(x):
    return tf.reshape(x, shape=[K.shape(x)[0], -1, K.shape(x)[-1]])