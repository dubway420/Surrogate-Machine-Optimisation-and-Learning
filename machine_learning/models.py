from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from collections.abc import Iterable
from tensorflow.keras import Model
from parmec_analysis.utils import is_in


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

    return activations


def convo_dense_layers(layers):
    """Takes a tuple or list - layers - and converts it into two tupes, one for convolutional layers - convo_layers -
    and one for dense layers.
    """

    dense_layers = ()
    dense_assigned = False

    # If the last item is iterable, it is the dense layers
    if iterable(layers[-1]):
        dense_layers = layers[-1]
        dense_assigned = True

    # If the first item in the layers argument is iterable, it is the convo layers
    if iterable(layers[0]):
        convo_layers = layers[0]

        # if dense_layers hasn't been set, then the last term isn't iterable. In this case, dense layers just becomes
        # everything after the first item
        if not dense_assigned:
            dense_layers = layers[1:]

    else:

        # if dense layers isn't set, it's likely that it's just a tuple of all numbers. In this case, the last two
        # numbers are used as the dense layers
        if not dense_assigned:
            convo_layers = layers[:-2]
            dense_layers = layers[-2:]

        # if the dense layers were the final item in the layers list, then everything up to that point is the convo
        # layers
        else:
            convo_layers = layers[:-1]

    return convo_layers, dense_layers


def kernels(number_convo_layers, kernel):
    if not iterable(kernel):

        kernels_list = [kernel for _ in range(number_convo_layers)]

    else:

        kernels_list = list(kernel)

        if len(kernels_list) < number_convo_layers:

            for _ in range(len(kernels_list), number_convo_layers):
                kernels_list.append(kernels_list[-1])

    return kernels_list


def dropout_validation(number_layers, fraction):
    if not iterable(fraction):

        kernels_list = [fraction for _ in range(number_layers)]

    else:

        kernels_list = list(fraction)

        if len(kernels_list) < number_layers:

            for _ in range(len(kernels_list), number_layers):
                kernels_list.append(kernels_list[-1])

    return kernels_list


class RegressionModels:

    # 0
    @staticmethod
    def multi_layer_perceptron(input_dims, output_dims, activation="linear", layers=(8, 4), dropout=0.5, regularizer=None):

        activations = layer_activations_validation(layers, activation)

        # define our MLP network
        model = Sequential()

        dropouts = dropout_validation(len(layers), dropout)

        # model.name = "Multi-layer Perceptron"

        model.add(Dense(layers[0], input_dim=input_dims, activity_regularizer=regularizer))
        model.add(Activation(activations[0]))

        if dropouts[0]:
            model.add(Dropout(dropouts[0]))


        layer_index = 1
        for layer_output_shape, act in zip(layers[1:], activations[1:]):
            model.add(Dense(layer_output_shape))
            model.add(Activation(act))

            if dropouts[layer_index]:
                model.add(Dropout(dropouts[layer_index]))

            layer_index += 1

        model.add(Dense(output_dims))
        model.add(Activation("linear"))

        # return our model
        return model


    # 3
    @staticmethod
    def convolutional_neural_network_2d(input_dims, output_dims, activation="linear", layers=(16, 32, 64),
                                        regularizer=None, dropout=0.5, kernel_shape=3, padding="valid"):

        convo_layers, dense_layers = convo_dense_layers(layers)

        layers_concatenated = convo_layers + dense_layers

        activations = layer_activations_validation(layers_concatenated, activation)

        filters = kernels(len(convo_layers), kernel_shape)

        dropouts = dropout_validation(len(layers_concatenated), dropout)

        # define our MLP network
        model = Sequential()

        # model.name = "Convolutional Neural Network"

        # Input layer
        model.add(Conv2D(convo_layers[0], kernel_size=filters[0], padding=padding, activity_regularizer=regularizer,
                         input_shape=input_dims, activation=activations[0]))

        if dropouts[0]:
            model.add(Dropout(dropouts[0]))

        layer_index = 1

        # Convolutional layers are added here
        for layer_output_shape in convo_layers[1:]:
            model.add(Conv2D(layer_output_shape, kernel_size=filters[layer_index], padding=padding,
                             activation=activations[layer_index]))

            if dropouts[layer_index]:
                model.add(Dropout(dropouts[layer_index]))

            layer_index += 1

        model.add(Flatten())

        # Dense layers are added here
        for layer_output_shape in dense_layers:
            model.add(Dense(layer_output_shape, activation=activations[layer_index]))

            if dropouts[layer_index]:
                model.add(Dropout(dropouts[layer_index]))

            layer_index += 1

        model.add(Dense(output_dims))
        # model.add(Activation("linear"))

        # return our model
        return model

        # 0

    # 4
    @staticmethod
    def existing(input_dims, output_dims, base_model=None, weights='imagenet', final_activation='linear',
                 final_bias=True):

        # If no model is specified, ResNet50 will be used
        if base_model is None:
            from tensorflow.keras.applications import ResNet50
            base_model = ResNet50(include_top=False, input_shape=input_dims, weights=weights)

        # Flatten the final layer's output
        x = Flatten()(base_model.output)

        # add the final (output) regression layer
        x = Dense(output_dims, activation=final_activation, use_bias=final_bias)(x)

        return Model(inputs=base_model.inputs, outputs=x)

    #5
    @staticmethod
    def vgg_model(input_dims, output_dims, base_model=None, weights='imagenet', final_bias=True, extra_layers=()):

        # If no model is specified, VGG16 will be used
        if base_model is None or is_in(base_model, 'vgg16'):
            from tensorflow.keras.applications import VGG16 as Base
        else:
            from tensorflow.keras.applications import VGG19 as Base

        base_model = Base(include_top=False, input_shape=input_dims, weights=weights)

        # Flatten the final layer's output
        x = Flatten()(base_model.output)

        for layer in extra_layers:
            x = layer(x)

        x = Dense(output_dims, activation="linear", use_bias=final_bias)(x)
        return Model(inputs=base_model.inputs, outputs=x)

