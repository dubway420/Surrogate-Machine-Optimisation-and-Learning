from keras.callbacks import Callback
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Flatten, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from parmec_analysis.dataset_generators import DatasetSingleFrame, Features1D, Labels
from parmec_analysis.utils import plot_names_title
from parmec_analysis.visualisation import CoreView, TrainingHistoryRealTime, model_comparison
import numpy as np


class SequentialBespoke(Sequential):
    """ Slight modification of sequential class to include short_name variable"""

    def __init__(self, short_name=""):
        super().__init__()
        self.short_name = short_name


class RegressionModels:

    @staticmethod
    def multi_layer_perceptron(input_dims, output_dims):

        # define our MLP network
        model = SequentialBespoke()

        model.name = "Multi-layer Perceptron"
        model.short_name = "MLP"

        model.add(Dense(8, input_dim=input_dims, activation="relu"))
        model.add(Dense(4, activation="relu"))
        model.add(Dense(output_dims, activation="linear"))

        # return our model
        return model

    @staticmethod
    def wider_model(input_dims, output_dims):

        # create model
        model = SequentialBespoke()

        model.name = "Wider Perceptron"
        model.short_name = "WP"

        model.add(Dense(20, input_dim=input_dims, kernel_initializer='normal', activation='relu'))
        model.add(Dense(4, activation="relu"))
        model.add(Dense(output_dims, kernel_initializer='normal'))

        return model

    @staticmethod
    def cnn1D(input_dims, output_dims):

        # create model
        model = SequentialBespoke()  # add model layers
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_dims))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dims, activation='softmax'))

        model.name = "CNN 1D"
        model.short_name = "CNN_1D"

        return model

    @staticmethod
    def cnn2D_type1(input_dims, output_dims):
        # create model
        model = SequentialBespoke()  # add model layers
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_dims))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dims, activation='softmax'))

        # Name the neural network
        model.name = "CNN 2D - 1"
        model.short_name = "CNN_2D_1"

        # model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def cnn_type2(width, height, depth, filters=(16, 32, 64), regress=True):

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
        model.name = "CNN 2D - 2"

        # return the CNN
        return model


class LossHistory(Callback):
    def __init__(self, loss_function, dataset, features, labels, iteration):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.loss_function = loss_function
        self.dataset = dataset
        self.features = features
        self.labels = labels
        self.iteration = iteration
        self.view = CoreView(self.model, dataset, features, labels, iteration)
        self.train_history = TrainingHistoryRealTime(dataset, features, labels, iteration, loss_function)

    def on_epoch_end(self, epoch, logs={}):
        self.train_history.update_data(logs, self.model)

        if epoch > 0 and (epoch % 10) == 0:
            self.view.update_data(epoch, self.model)


# Lists the functions of the models
models = [getattr(RegressionModels(), string_method) for string_method in RegressionModels().__dir__()[1:-25]]

results_path = "/media/huw/Disk1/parmec_results/"

no_instances = 80

# Features
channels_features = 'all'
levels_features = 'all'
array_type = 'Positions Only'

# Labels
channels_labels = 'all'
levels_labels = 'all'
result_type = 'all'
result_time = 48
result_column = 1

# Machine learning
opt = Adam(lr=1e-3, decay=1e-3 / 200)
loss = mean_absolute_percentage_error
repeat_each_model = 2
epochs = 50

###########################################################
# ################## Dataset ################################
###########################################################

dataset_80 = DatasetSingleFrame(results_path, number_of_cases=no_instances)

features_1d = Features1D(dataset_80)

labels_48_all = Labels(dataset_80, result_time=48, result_type='all')

###########################################################
# ################## Model Setup ###########################
###########################################################

model_short_names = []

models_compiled = []
model_histories = []

losses_training = []
losses_validation = []

for model_no in range(2):

    models_type = []
    models_type_histories = []

    model_losses_training = []
    model_losses_validation = []

    for i in range(1, repeat_each_model + 1):
        history = LossHistory(loss, dataset_80, features_1d, labels_48_all, i)
        models_type.append(models[model_no](features_1d.feature_shape, labels_48_all.label_shape))
        models_type[-1].compile(loss=loss.__name__, optimizer=opt)
        model_fit = models_type[-1].fit(features_1d.training_set(), labels_48_all.training_set(),
                                        validation_data=(features_1d.validation_set(),
                                                         labels_48_all.validation_set()),
                                        epochs=epochs, verbose=0, callbacks=[history])
        models_type_histories.append(model_fit)
        model_losses_training.append(model_fit.history['loss'][-1])
        model_losses_validation.append(model_fit.history['val_loss'][-1])

    model_short_names.append(models_type[-1].short_name)

    models_compiled.append(models_type)
    model_histories.append(models_type_histories)

    losses_training.append(model_losses_training)
    losses_validation.append(model_losses_validation)

losses_mean_training = np.mean(losses_training, axis=1)
losses_mean_validation = np.mean(losses_validation, axis=1)


model_comparison(model_short_names, losses_mean_training, losses_mean_validation)
