from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Flatten
from keras.models import Sequential
from collections.abc import Iterable


def iterable(obj):
    if isinstance(obj, Iterable):

        if isinstance(obj, str):
            return False

        return True

    return False


def layer_activations_validation(layer_output_shapes, activation):
    """ Takes inputs for the layer sizes and activations, and returns two lists of the same size"""

    if not iterable(layer_output_shapes):
        layer_output_shapes = [layer_output_shapes, ]

    if not iterable(activation):

        activations = [activation for _ in range(len(layer_output_shapes))]

    else:

        activations = list(activation)

        if len(activations) < len(layer_output_shapes):
            for _ in layer_output_shapes[len(activations):]:
                activations.append(activations[-1])

    return layer_output_shapes, activations
    
    
def kernels(number_convo_layers, kernel):

    if not iterable(kernel):

        kernels_list = [kernel for _ in range(number_convo_layers)]

    else:

        kernels_list = list(kernel)

        if len(kernels_list) < number_convo_layers:

            for _ in range(len(kernels_list), number_convo_layers):

                kernels_list.append(kernels_list[-1])

    return kernels_list    


class RegressionModels:

    # 0
    @staticmethod
    def multi_layer_perceptron(input_dims, output_dims, activation="linear", layers=(8, 4), regularizer=None):

        layer_output_shapes, activations = layer_activations_validation(layers, activation)

        # define our MLP network
        model = Sequential()

        model.name = "Multi-layer Perceptron"

        model.add(Dense(layer_output_shapes[0], input_dim=input_dims, activity_regularizer=regularizer))
        model.add(Activation(activations[0]))

        for layer_output_shape, act in zip(layer_output_shapes[1:], activations[1:]):
            model.add(Dense(layer_output_shape))
            model.add(Activation(act))

        model.add(Dense(output_dims))
        model.add(Activation("linear"))

        # return our model
        return model

    # 1
    @staticmethod
    def wider_model(input_dims, output_dims):

        # create model
        model = Sequential()

        model.name = "Wider Perceptron"

        model.add(Dense(20, input_dim=input_dims, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(4, activation="tanh"))
        model.add(Dense(output_dims, kernel_initializer='normal'))
        # Regularisation

        return model

    # 2
    @staticmethod
    def cnn1D(input_dims, output_dims):

        # create model
        model = Sequential()  # add model layers
        model.add(Conv1D(64, kernel_size=3, activation='tanh', input_shape=input_dims))
        model.add(Conv1D(32, kernel_size=3, activation='tanh'))
        model.add(Flatten())
        model.add(Dense(output_dims, activation='tanh'))

        model.name = "CNN 1D"

        return model

    @staticmethod
    def cnn1D_k6(input_dims, output_dims):

        # create model
        model = Sequential()  # add model layers
        model.add(Conv1D(64, kernel_size=6, activation='tanh', input_shape=input_dims))
        model.add(Conv1D(32, kernel_size=3, activation='tanh'))
        model.add(Flatten())
        model.add(Dense(output_dims, activation='tanh'))

        model.name = "CNN 1D"

        return model

    # 3
    @staticmethod
    def cnn2D_type1(input_dims, output_dims):
        # create model
        model = Sequential()  # add model layers
        model.add(Conv2D(64, kernel_size=3, activation='tanh', input_shape=input_dims))
        model.add(Conv2D(32, kernel_size=3, activation='tanh'))
        model.add(Flatten())
        model.add(Dense(output_dims, activation='tanh'))

        # Name the neural network
        model.name = "CNN 2D - 1"

        # model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # 4
    @staticmethod
    def cnn2D_type2(input_dims, output_dims, filters=(16, 32, 64), activation="tanh", regress=True):

        model = Sequential()  # add model layers
        model.add(Conv2D(filters[0], kernel_size=3, activation=activation, input_shape=input_dims))

        if len(filters) > 1:
            for filter_dims in filters[1:]:
                model.add(Conv2D(filter_dims, kernel_size=3, activation=activation))
                model.add(BatchNormalization())  # axis=chan_dim

        model.add(Flatten())
        model.add(Dense(16, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='tanh'))

        # check to see if the regression node should be added
        if regress:
            model.add(Dense(output_dims, activation="linear"))

        # Name the neural network
        model.name = "CNN 2D - 2"

        # return the CNN
        return model

    # 0
    @staticmethod
    def convolutional_neural_network_2d(input_dims, output_dims, activation="linear", layers=(16, 32, 64),
                                        regularizer=None, dropout=0.5, kernel_shape=3, padding="valid"):

        layer_output_shapes, activations = layer_activations_validation(layers, activation)

        # define our MLP network
        model = Sequential()

        model.name = "Convolutional Neural Network"

        model.add(Conv2D(layer_output_shapes[0], kernel_size=kernel_shape, input_shape=input_dims,
                         activity_regularizer=regularizer, padding=padding))
        model.add(Activation(activations[0]))

        for layer_output_shape, act in zip(layer_output_shapes[1:-2], activations[1:-2]):
            model.add(Conv2D(layer_output_shape, kernel_size=kernel_shape, padding=padding))
            model.add(Activation(act))

        model.add(Flatten())

        model.add(Dense(layer_output_shapes[-2], activation=activations[-2]))
        model.add(Dropout(dropout))
        model.add(Dense(layer_output_shapes[-1], activation=activations[-1]))

        model.add(Dense(output_dims))
        model.add(Activation("linear"))

        # return our model
        return model
        
        # 0
    @staticmethod
    def convolutional_neural_network_2d_no_fc(input_dims, output_dims, activation="linear", layers=(16, 32, 64),
                                        regularizer=None, dropout=0.5, kernel_shape=3):

        layer_output_shapes, activations = layer_activations_validation(layers, activation)

        # define our MLP network
        model = Sequential()

        model.name = "Convolutional Neural Network"

        model.add(Conv2D(layer_output_shapes[0], kernel_size=kernel_shape, input_shape=input_dims,
                         activity_regularizer=regularizer))
        model.add(Activation(activations[0]))

        for layer_output_shape, act in zip(layer_output_shapes[1:-2], activations[1:-2]):
            model.add(Conv2D(layer_output_shape, kernel_size=kernel_shape))
            model.add(Activation(act))

        model.add(Flatten())

        model.add(Dense(layer_output_shapes[-2], activation=activations[-2]))
        model.add(Dropout(dropout))
        model.add(Dense(layer_output_shapes[-1], activation=activations[-1]))

        model.add(Dense(output_dims))
        model.add(Activation("linear"))

        # return our model
        return model    
