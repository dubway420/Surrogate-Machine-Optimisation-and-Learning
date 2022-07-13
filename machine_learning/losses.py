import keras.backend as K
import tensorflow as tf
from scipy.stats import norm 

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

    adjustment_factor = norm.pdf(i, modal_bin, std)/norm_max
    
    if square:
        adjustment_factor = adjustment_factor**2
    

    return adjustment_factor + 1

def mse_norm_adjusted(modal_bin, std, norm_max, square=True, mean=False):


    def inner_method(y_true, y_pred):

        adj = adjustment_factor(y_true, modal_bin, std, norm_max, square)

        # calculating squared difference between target and predicted values
        loss = K.square(y_pred - y_true)

        adj_loss = loss * adj

        if mean:
            adj_loss = K.mean(adj_loss)
            
        return adj_loss

    return inner_method
    