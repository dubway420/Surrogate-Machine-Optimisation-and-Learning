import keras.backend as K
import tensorflow as tf

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


def adjusted_mse(average):
    def inner_method(y_true, y_pred):
        # calculating squared difference between target and predicted values
        loss = K.square(y_pred - y_true)

        # Calculates an array of adjustment values. The value will depend on the distance between the mean
        extremeness_adjust = 1 + K.abs(y_true - average)

        # multiplying the values with weights along batch dimension
        loss = loss * extremeness_adjust  # (batch_size, 2)

        # summing both loss values along batch dimension
        loss = K.sum(loss)  # (batch_size,)

        return loss

    return inner_method
