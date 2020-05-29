from keras.callbacks import Callback
import tensorflow as tf
from keras import backend as K
from parmec_analysis.utils import folder_validation, experiment_iteration, save_results, plot_names_title
from parmec_analysis.visualisation import CoreView, TrainingHistoryRealTime
from machine_learning.experiment_summary import summary
import os
import numpy as np
import matplotlib.pyplot as plt


class LossHistory(Callback):
    def __init__(self, loss_function, trial, experiment, iteration, print_every_n_epochs=None):
        super().__init__()
        self.losses = []
        self.val_losses = []

        self.loss_function = loss_function
        self.trial = trial
        self.experiment = experiment
        self.iteration = iteration

        self.features_training = self.experiment.features.training_set()
        self.features_validation = self.experiment.features.validation_set()

        labels_training_flat = self.experiment.labels.training_set().flatten()
        labels_validation_flat = self.experiment.labels.validation_set().flatten()

        # gets some random indices of the flattened training and validation lists
        self.labels_training_sample_indices = np.random.choice(len(labels_training_flat), 10000)
        self.labels_validation_sample_indices = np.random.choice(len(labels_validation_flat), 10000)

        self.sample_labels_training = [labels_training_flat[i] for i in self.labels_training_sample_indices]
        self.sample_labels_validation = [labels_validation_flat[i] for i in self.labels_validation_sample_indices]

        # placeholders for interval predictions
        self.interval_predictions_training = []
        self.interval_predictions_validation = []

        if not print_every_n_epochs:
            print_every_n_epochs = 1000

        self.plot_every_n_epochs = print_every_n_epochs

        self.train_history = TrainingHistoryRealTime(trial, iteration, experiment, loss_function,
                                                     self.plot_every_n_epochs)

        iteration_l = str(iteration) + "L"

        self.train_history_later = TrainingHistoryRealTime(trial, iteration_l, experiment, loss_function,
                                                           (3 * self.plot_every_n_epochs))

        self.view = CoreView(trial, iteration, experiment)

        self.epochs_with_results = []

    def on_epoch_end(self, epoch, logs={}):

        epoch_p1 = epoch + 1

        self.train_history.update_data(logs, self.model, plot=False)
        self.train_history_later.update_data(logs, self.model, plot=False)

        if (epoch_p1 % self.plot_every_n_epochs) == 0:

            ###############################################################################

            self.epochs_with_results.append(epoch_p1)

            predictions_training_flat = self.model.predict(self.features_training).flatten()
            predictions_validation_flat = self.model.predict(self.features_validation).flatten()

            sample_predictions_train = [predictions_training_flat[i] for i in self.labels_training_sample_indices]
            sample_predictions_val = [predictions_validation_flat[i] for i in self.labels_validation_sample_indices]

            self.interval_predictions_training.append(sample_predictions_train)
            self.interval_predictions_validation.append(sample_predictions_val)

            fig, axes = plt.subplots(len(self.interval_predictions_training), 2, sharex=True)

            if len(self.interval_predictions_training) == 1:
                axes = [axes, ]

            for i, row in enumerate(axes):
                total_values_train = np.array([self.sample_labels_training, self.interval_predictions_training[i]])

                min_vals_train = np.amin(total_values_train) * 0.8
                max_vals_train = np.amax(total_values_train) * 1.2

                row[0].scatter(self.sample_labels_training, self.interval_predictions_training[i])
                row[0].plot([min_vals_train, max_vals_train], [min_vals_train, max_vals_train], c='yellow')
                row[0].set_xlim([min_vals_train, max_vals_train])
                row[0].set_ylim([min_vals_train, max_vals_train])

                row[0].set_ylabel(("Epoch: " + str(self.epochs_with_results[i])))

                total_values_valid = np.array([self.sample_labels_validation, self.interval_predictions_validation[i]])
                min_vals_valid = np.amin(total_values_valid) * 0.8
                max_vals_valid = np.amax(total_values_valid) * 1.2

                row[1].scatter(self.sample_labels_validation, self.interval_predictions_validation[i])
                row[1].plot([min_vals_valid, max_vals_valid], [min_vals_valid, max_vals_valid], c='yellow')
                row[1].set_xlim([min_vals_valid, max_vals_valid])
                row[1].set_ylim([min_vals_valid, max_vals_valid])

            _, file_name = plot_names_title(self.experiment, self.iteration)

            file_name = self.trial + "/" + self.experiment.name + "/correlation_" + file_name
            plt.savefig(file_name)
            plt.close()
            ###############################################################################

            self.train_history.plotting()
            self.train_history_later.plotting()
            self.view.update_data(epoch, self.model, True, True, False)


NUMCORES = int(os.getenv("NSLOTS", 1))


# sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
#                                         allow_soft_placement=True,
#                                         device_count={'CPU': NUMCORES}))
#
# # Set the Keras TF session
# K.set_session(sess)


def run_experiment(experiment):
    trial_name = experiment.trial.trial_name

    folder_validation(trial_name)

    experiment_folder = trial_name + "/" + experiment.name
    folder_validation(experiment_folder)

    summary(experiment)

    exp_i = experiment_iteration(experiment.name, trial_name)

    loss = experiment.trial.loss_function
    history = LossHistory(loss, trial_name, experiment, exp_i, experiment.trial.plot_every_n_epochs)

    model_i = experiment.model

    model_i.compile(loss=loss.__name__, optimizer=experiment.trial.optimiser)

    model_fit = model_i.fit(experiment.features.training_set(), experiment.labels.training_set(),
                            validation_data=(experiment.features.validation_set(),
                                             experiment.labels.validation_set()),
                            epochs=experiment.trial.epochs, verbose=0, callbacks=[history])

    losses = [model_fit.history['loss'][-1], model_fit.history['val_loss'][-1]]

    save_results(experiment.name, trial_name, exp_i, losses)
