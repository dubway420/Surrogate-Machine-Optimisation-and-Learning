import keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math as m

tfd = tfp.distributions

'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


'''
 ' Same as above but returns the mean loss.
'''


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def adjusted_mse(average, alpha=1, square_adjust=False):
    def inner_method(y_true, y_pred):
        # calculating squared difference between target and predicted values
        loss = K.square(y_pred - y_true)

        # Calculates an array of adjustment values. The value will depend on the distance between the mean
        extremeness_adjust = 1 + K.abs(y_true - average)*alpha

        # If square_adjust=True, we square the extremeness_adjust value
        if square_adjust:
            extremeness_adjust *= extremeness_adjust

        # multiplying the values with weights along batch dimension
        loss = loss * extremeness_adjust  # (batch_size, 2)

        # summing both loss values along batch dimension
        loss = K.sum(loss)  # (batch_size,)

        return loss

    return inner_method


def adjustment_factor(i, modal_bin, std, norm_max, square=True):
    
    #convert i to 64 bit float
    i = tf.cast(i, tf.float64)
    std = tf.cast(std, tf.float64)
    modal_bin = tf.cast(modal_bin, tf.float64)

    rhs = tf.exp(tf.multiply(tf.constant(-0.5, dtype=tf.float64), tf.pow(tf.divide(i - modal_bin, std), tf.constant(2, dtype=tf.float64))))

    lhs = tf.divide(1, tf.multiply(std, tf.sqrt(tf.multiply(tf.constant(2.0, dtype=tf.float64), tf.constant(m.pi, dtype=tf.float64)))))

    lhs = tf.cast(lhs, tf.float64)
    # tf.divide(
    adjustment_factor = tf.divide(tf.multiply(lhs, rhs), tf.constant(norm_max))

    # adjustment_factor = tfd.Normal(modal_bin, std).prob(i) / norm_max
    
    if square:
        adjustment_factor = adjustment_factor**2
    

    return adjustment_factor + 1


def mse_norm_adjusted(modal_bin, std, norm_max, square=True, mean=False):


    def inner_method(y_true, y_pred):

        adj = adjustment_factor(y_pred, modal_bin, std, norm_max, square)

        # calculating squared difference between target and predicted values
        loss = K.square(y_pred - y_true)

        # convert loss to 64 bit float
        loss = tf.cast(loss, tf.float64)

        adj_loss = loss * adj

        if mean:
            adj_loss = K.mean(adj_loss)
            
        return adj_loss

    return inner_method
