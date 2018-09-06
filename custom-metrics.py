from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense
import tensorflow as tf


def cumulative_accuracy(y_true, y_pred):
    """ This metrics is used to evaluate regression like classification
        Formula:
            acc = K_n/K*100
            K_n: A number of values which are smaller than n
            K: A number of given values
    """
    m = len(y_true)
    n = 3.0 # threshold
    return tf.reduce_sum(y_pred<n)/m


    