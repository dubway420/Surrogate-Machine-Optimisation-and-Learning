# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model


def multi_layer_perceptron(input_dims, output_dims):
    # define our MLP network
    model = Sequential()

    model.name = "Multi-layer Perceptron"

    model.add(Dense(8, input_dim=input_dims, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_dims, activation="linear"))

    # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')

    # return our model
    return model


def wider_model(input_dims, output_dims):
    # create model
    model = Sequential()

    model.name = "Wider Perceptron"

    model.add(Dense(20, input_dim=input_dims, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_dims, kernel_initializer='normal'))

    # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def cnn1D(input_dims, output_dims):
    # create model
    model = Sequential()  # add model layers
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_dims))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dims, activation='softmax'))

    # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def cnn2D(input_dims, output_dims):
    # create model
    model = Sequential()  # add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_dims))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dims, activation='softmax'))

    # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):

    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    input_shape = (height, width, depth)
    chan_dim = -1

    # define the model input
    inputs = Input(shape=input_shape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chan_dim)(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(321, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # Name the neural network
    model.name = "CNN 2"

    # return the CNN
    return model