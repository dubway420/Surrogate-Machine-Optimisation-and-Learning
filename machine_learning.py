# from keras.layers.convolutional import Conv1D, Conv2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers import Flatten, Input
# from keras.models import Model, Sequential
# from keras.optimizers import Adam
from parmec_analysis.utils import cases_list, is_in, split_separators
from parmec_analysis import reactor_case
from parmec_analysis.dataset_generators import features_and_labels_single_frame
import numpy as np

# class RegressionModels:
#
#     @staticmethod
#     def multi_layer_perceptron(input_dims, output_dims):
#
#         # define our MLP network
#         model = Sequential()
#
#         model.name = "Multi-layer Perceptron"
#
#         model.add(Dense(8, input_dim=input_dims, activation="relu"))
#         model.add(Dense(4, activation="relu"))
#         model.add(Dense(output_dims, activation="linear"))
#
#         # return our model
#         return model
#
#     @staticmethod
#     def wider_model(input_dims, output_dims):
#
#         # create model
#         model = Sequential()
#
#         model.name = "Wider Perceptron"
#
#         model.add(Dense(20, input_dim=input_dims, kernel_initializer='normal', activation='relu'))
#         model.add(Dense(4, activation="relu"))
#         model.add(Dense(output_dims, kernel_initializer='normal'))
#
#         return model
#
#     @staticmethod
#     def cnn1D(input_dims, output_dims):
#
#         # create model
#         model = Sequential()  # add model layers
#         model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_dims))
#         model.add(Conv1D(32, kernel_size=3, activation='relu'))
#         model.add(Flatten())
#         model.add(Dense(output_dims, activation='softmax'))
#
#         model.name = "CNN 1D"
#
#         return model
#
#     @staticmethod
#     def cnn2D_type1(input_dims, output_dims):
#         # create model
#         model = Sequential()  # add model layers
#         model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_dims))
#         model.add(Conv2D(32, kernel_size=3, activation='relu'))
#         model.add(Flatten())
#         model.add(Dense(output_dims, activation='softmax'))
#
#         # Name the neural network
#         model.name = "CNN 2D - 1"
#
#         # model.compile(loss='mean_squared_error', optimizer='adam')
#         return model
#
#     @staticmethod
#     def cnn_type2(width, height, depth, filters=(16, 32, 64), regress=True):
#
#         # initialize the input shape and channel dimension, assuming
#         # TensorFlow/channels-last ordering
#         input_shape = (height, width, depth)
#         chan_dim = -1
#
#         # define the model input
#         inputs = Input(shape=input_shape)
#
#         # loop over the number of filters
#         for (i, f) in enumerate(filters):
#             # if this is the first CONV layer then set the input
#             # appropriately
#             if i == 0:
#                 x = inputs
#
#             # CONV => RELU => BN => POOL
#             x = Conv2D(f, (3, 3), padding="same")(x)
#             x = Activation("relu")(x)
#             x = BatchNormalization(axis=chan_dim)(x)
#             # x = MaxPooling2D(pool_size=(2, 2))(x)
#
#         # flatten the volume, then FC => RELU => BN => DROPOUT
#         x = Flatten()(x)
#         x = Dense(16)(x)
#         x = Activation("relu")(x)
#         x = BatchNormalization(axis=chan_dim)(x)
#         x = Dropout(0.5)(x)
#
#         # apply another FC layer, this one to match the number of nodes
#         # coming out of the MLP
#         x = Dense(4)(x)
#         x = Activation("relu")(x)
#
#         # check to see if the regression node should be added
#         if regress:
#             x = Dense(321, activation="linear")(x)
#
#         # construct the CNN
#         model = Model(inputs, x)
#
#         # Name the neural network
#         model.name = "CNN 2D - 2"
#
#         # return the CNN
#         return model


# Lists the functions of the models
# models = [getattr(RegressionModels(), string_method) for string_method in RegressionModels().__dir__()[1:-25]]

# Get the coordinates of the interstitial channels
# case_intact = 'intact_core_rb'
# instance_intact = reactor_case.Parse(case_intact)
# channel_coord_list_inter = instance_intact.get_brick_xyz_positions('xy', channel_type='inter')

features = features_and_labels_single_frame("/media/huw/Disk1/parmec_results/", x_type='ori',
                                            result_time=48, result_type='sum', flat_y=True,
                                            features_labels='feat', x_request='3d',
                                            levels='all')











