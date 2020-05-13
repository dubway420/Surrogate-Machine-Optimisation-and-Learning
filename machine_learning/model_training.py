from keras.callbacks import Callback
import tensorflow as tf
from keras import backend as K
from parmec_analysis.utils import folder_validation, experiment_iteration, save_results
from parmec_analysis.visualisation import CoreView, TrainingHistoryRealTime
from machine_learning.experiment_summary import summary
import os


class LossHistory(Callback):
    def __init__(self, loss_function, trial, experiment, iteration, print_every_n_epochs=None):
        super().__init__()
        self.losses = []
        self.val_losses = []

        self.loss_function = loss_function
        self.trial = trial
        self.experiment = experiment
        self.iteration = iteration

        if not print_every_n_epochs:
            print_every_n_epochs = 1000

        self.plot_every_n_epochs = print_every_n_epochs

        self.view = CoreView(trial, iteration, experiment)
        self.train_history = TrainingHistoryRealTime(trial, iteration, experiment, loss_function,
                                                     self.plot_every_n_epochs)

        iteration_l = str(iteration) + "L"

        self.train_history_later = TrainingHistoryRealTime(trial, iteration_l, experiment, loss_function,
                                                           (3 * self.plot_every_n_epochs))

    def on_epoch_end(self, epoch, logs={}):

        epoch_p1 = epoch + 1

        self.train_history.update_data(logs, self.model, plot=False)
        self.train_history_later.update_data(logs, self.model, plot=False)

        if (epoch_p1 % self.plot_every_n_epochs) == 0:
            self.view.update_data(epoch, self.model)
            self.train_history.plotting()
            self.train_history_later.plotting()


NUMCORES = int(os.getenv("NSLOTS", 1))

sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
                                        allow_soft_placement=True,
                                        device_count={'CPU': NUMCORES}))

# Set the Keras TF session
K.set_session(sess)


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
