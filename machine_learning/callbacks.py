import matplotlib as mpl
# Agg backend will render without X server on a compute node in batch
mpl.use('Agg')
import numpy as np
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from parmec_analysis.utils import plot_names_title
from parmec_analysis.visualisation import CoreView, TrainingHistoryRealTime



class LossHistory(Callback):
    def __init__(self, loss_function, trial, experiment, iteration, print_every_n_epochs=None, channel_to_plot=161):
        super().__init__()
        self.losses = []
        self.val_losses = []

        self.cases = [34, 78, 127]

        self.loss_function = loss_function
        self.trial = trial
        self.experiment = experiment
        self.iteration = iteration

        self.features_training = self.experiment.features.training_set()
        self.features_validation = self.experiment.features.validation_set()

        self.labels_training = self.experiment.labels.training_set()
        self.labels_validation = self.experiment.labels.validation_set()

        self.labels_training_select = [self.labels_training[i] for i in self.cases]
        self.labels_validation_select = [self.labels_validation[i] for i in self.cases]

        self.predictions_training_select = []
        self.predictions_validation_select = []

        labels_training_flat = self.labels_training.flatten()
        labels_validation_flat = self.labels_validation.flatten()

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

        if experiment.labels.channels is 'all':
            self.view = CoreView(trial, iteration, experiment)
            
            if experiment.labels.levels is 'all':
               self.view5 = CoreView(trial, iteration, experiment, convert_to="5")
               self.view7 = CoreView(trial, iteration, experiment, convert_to="7")
               self.view10 = CoreView(trial, iteration, experiment, convert_to="10")
               self.view12 = CoreView(trial, iteration, experiment, convert_to="12")

        self.epochs_with_results = []

        self.channel_to_plot = channel_to_plot

    def on_epoch_end(self, epoch, logs={}):

        epoch_p1 = epoch + 1

        self.train_history.update_data(logs, self.model, plot=False)
        self.train_history_later.update_data(logs, self.model, plot=False)

        if (epoch_p1 % self.plot_every_n_epochs) == 0 or epoch_p1 == self.experiment.trial.epochs:

            self.epochs_with_results.append(epoch_p1)

            ###############################################################################

            predictions_training = self.model.predict(self.features_training)
            predictions_validation = self.model.predict(self.features_validation)

            predictions_training_flat = predictions_training.flatten()
            predictions_validation_flat = predictions_validation.flatten()

            ###############################################################################

            sample_predictions_train = [predictions_training_flat[i] for i in self.labels_training_sample_indices]
            sample_predictions_val = [predictions_validation_flat[i] for i in self.labels_validation_sample_indices]

            self.interval_predictions_training.append(sample_predictions_train)
            self.interval_predictions_validation.append(sample_predictions_val)

            ###############################################################################

            fig, axes = plt.subplots(len(self.interval_predictions_training), 2, sharex=True)

            if len(self.interval_predictions_training) == 1:
                axes = [axes, ]

            for i, row in enumerate(axes):

                if i == 0:
                    row[0].set_title("Training")
                    row[1].set_title("Validation")

                total_values_train = np.array([self.sample_labels_training, self.interval_predictions_training[i]])

                min_vals_train = np.amin(total_values_train) * 0.8
                max_vals_train = np.amax(total_values_train) * 1.2

                row[0].scatter(self.sample_labels_training, self.interval_predictions_training[i])
                row[0].plot([min_vals_train, max_vals_train], [min_vals_train, max_vals_train], c='yellow')
                row[0].set_xlim([min_vals_train, max_vals_train])
                row[0].set_ylim([min_vals_train, max_vals_train])
                row[0].set_xticks([])
                row[0].set_yticks([])
                row[0].set_aspect('equal')

                row[0].set_ylabel(("Epoch: " + str(self.epochs_with_results[i])))

                total_values_valid = np.array([self.sample_labels_validation, self.interval_predictions_validation[i]])
                min_vals_valid = np.amin(total_values_valid) * 0.8
                max_vals_valid = np.amax(total_values_valid) * 1.2

                row[1].scatter(self.sample_labels_validation, self.interval_predictions_validation[i])
                row[1].plot([min_vals_valid, max_vals_valid], [min_vals_valid, max_vals_valid], c='yellow')
                row[1].set_xlim([min_vals_valid, max_vals_valid])
                row[1].set_ylim([min_vals_valid, max_vals_valid])
                row[1].set_xticks([])
                row[1].set_yticks([])
                row[1].set_aspect('equal')

            _, file_name = plot_names_title(self.experiment, self.iteration)
            
            #plt.subplots_adjust(wspace=0, hspace=0)
            #fig.tight_layout()
            
            file_name = self.trial + "/" + self.experiment.name + "/correlation_" + file_name
            plt.savefig(file_name, bbox_inches='tight')
            plt.close()
            ###############################################################################

            self.predictions_training_select.append([predictions_training[i] for i in self.cases])

            # Training plots
            fig, axes = plt.subplots(len(self.predictions_training_select) + 1, len(self.cases))

            ground_truth_row = axes[0]

            ground_truth_row[0].set_ylabel("GT")

            # These values allow the slicing of the label arrays to get the result for the channel to plot
            start_channel = (self.channel_to_plot-1) * self.experiment.labels.number_levels
            end_channel = start_channel + self.experiment.labels.number_levels + 1

            for i, ax in enumerate(ground_truth_row):
                case_title = self.experiment.dataset.training_instances()[self.cases[i]].get_id()
                ax.set_title(case_title)
                ground_truth = self.labels_training_select[i][start_channel:end_channel]
                ax.barh(np.arange(len(ground_truth)), ground_truth)

            for i, row in enumerate(axes[1:]):

                row_title = "E" + str(self.epochs_with_results[i])
                row[0].set_ylabel(row_title)

                for j, ax in enumerate(row):
                    case_result = self.predictions_training_select[i][j][start_channel:end_channel]

                    ax.barh(np.arange(len(case_result)), case_result)

            _, file_name = plot_names_title(self.experiment, self.iteration)

            file_name = self.trial + "/" + self.experiment.name + "/training_channelview_" + file_name
            plt.savefig(file_name)
            plt.close()

            # Validation plots
            self.predictions_validation_select.append([predictions_validation[i] for i in self.cases])

            fig, axes = plt.subplots(len(self.predictions_validation_select) + 1, len(self.cases))

            ground_truth_row = axes[0]

            ground_truth_row[0].set_ylabel("GT")

            for i, ax in enumerate(ground_truth_row):
                case_title = self.experiment.dataset.validation_instances()[self.cases[i]].get_id()
                ax.set_title(case_title)
                ax.barh(np.arange(len(self.labels_validation_select[i])), self.labels_validation_select[i])

            for i, row in enumerate(axes[1:]):

                row_title = "E" + str(self.epochs_with_results[i])
                row[0].set_ylabel(row_title)

                for j, ax in enumerate(row):
                    case_result = self.predictions_validation_select[i][j]

                    ax.barh(np.arange(len(case_result)), case_result)

            # fig.tight_layout()

            _, file_name = plot_names_title(self.experiment, self.iteration)

            file_name = self.trial + "/" + self.experiment.name + "/validation_channelview_" + file_name
            plt.savefig(file_name)
            plt.close()

            ###############################################################################

            self.train_history.plotting()
            #self.train_history_later.plotting()

            if self.experiment.labels.channels is 'all':
                self.view.update_data(epoch, self.model, True, False, False)

                if self.experiment.labels.levels is 'all':
                    self.view5.update_data(epoch, self.model, True, False, False)
                    self.view7.update_data(epoch, self.model, True, False, False)
                    self.view10.update_data(epoch, self.model, True, False, False)
                    self.view12.update_data(epoch, self.model, True, False, False)

        if epoch_p1 == self.experiment.trial.epochs:
        
            # Training Histogram
            sns.set(style="darkgrid")
            sns.distplot(self.labels_training.flatten(), color='green', label="Ground Truth Labels")

            sns.distplot(predictions_training_flat, color='red', label="Predictions Final Epoch")
            # plt.title('Url Length Distribution')
            plt.legend(loc='upper right')
            plt.xlabel('Displacement (mm)')
            _, file_name = plot_names_title(self.experiment, self.iteration)

            file_name = self.trial + "/" + self.experiment.name + "/training_histogram_" + file_name
            plt.savefig(file_name)
            plt.close()
            
            # Validation Histogram
            sns.set(style="darkgrid")
            sns.distplot(self.labels_validation.flatten(), color='green', label="Ground Truth Labels")

            sns.distplot(predictions_validation_flat, color='red', label="Predictions Final Epoch")
            # plt.title('Url Length Distribution')
            plt.legend(loc='upper right')
            plt.xlabel('Displacement (mm)')
            _, file_name = plot_names_title(self.experiment, self.iteration)

            file_name = self.trial + "/" + self.experiment.name + "/validation_histogram_" + file_name
            plt.savefig(file_name)
            plt.close()


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 110
    decay_step_2 = 310

    if epoch == decay_step or epoch == decay_step_2:
        return lr * decay_rate

    return lr

