import matplotlib as mpl
# Agg backend will render without X server on a compute node in batch
mpl.use('Agg')
import numpy as np
from matplotlib import mlab
# from sympy.stats import density
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from parmec_analysis.utils import plot_names_title, is_in
from parmec_analysis.visualisation import CoreView, TrainingHistoryRealTime, autolabel

colours = ['b', 'g', 'y', 'r', 'c', 'm', 'lime', 'darkorange']


def histo_fit(y, bins):
    mu = np.mean(y)
    sigma = np.std(y)

    # add a 'best fit' line
    best_fit_line = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                     np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))

    return best_fit_line


def instance_chooser(labels_set, type_chosen="rand"):
    if is_in(type_chosen, "min"):
        index = np.argmin(labels_set)
    elif is_in(type_chosen, "max"):
        index = np.argmax(labels_set)
    elif is_in(type_chosen, "avg") or is_in(type_chosen, "med"):
        index = np.argsort(labels_set[:, 0])[int(len(labels_set) / 2)]
    else:
        index = np.random.randint(0, len(labels_set))

    val = labels_set[index, 0]
    # name = data_set[index].split('/')[-1]

    return {"value": val, "index": index}


def plotter_foursquare(x, y, labels_set, prediction_set, fig_title):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # ====================================
    # Loss over time plot
    # ====================================

    c = colours[-len(x):]

    ax1.plot(x, y, c='black')

    ax1.scatter(x, y,
                c=c)

    ax1.plot([x[0], x[-1]],
             [y[-1], y[-1]],
             "k:")

    ax1.set(xlabel='Epoch', ylabel='Loss')

    xticks = [int(i) for i in x]

    ax1.set_xticks(xticks)
    ax1.set_yticks(y)

    # ====================================
    # Correlation plot and Histogram
    # ====================================

    n, bins, patches = ax3.hist(labels_set.flatten(), color='grey', label="Ground Truth",
                                density=True)

    y = histo_fit(labels_set.flatten(), bins)

    ax3.plot(bins, y, '--', c='black')

    # Draw a straight line from 0 to 1 denoting a 'perfect' result
    ax2.plot([0, 1.1], 'k:', c='black', alpha=0.5)
    ax2.plot([0, 1], c='black')
    ax2.plot([0, 0.9], 'k:', c='black', alpha=0.5)

    epoch_labels = ["Ground Truth"]

    for predictions, epoch, col in zip(prediction_set, x, c):
        label = "Epoch " + str(epoch)
        epoch_labels.append(label)

        ax2.scatter(labels_set, predictions, label=label, c=col)

        m, b = np.polyfit(labels_set.flatten(), predictions, 1)
        #
        ax2.plot(labels_set, m * labels_set + b, c=col)

        n, bins, patches = ax3.hist(predictions, color=col, label=label, density=True, alpha=0.5)

        y = histo_fit(predictions, bins)

        ax3.plot(bins, y, '--', c=col)

    ax2.legend()
    ax2.set(xlabel='Ground Truth', ylabel='Predictions')

    ax3.legend()

    # ====================================
    # Bar Graph
    # ====================================

    instance_types = ["min", "avg", "max"]
    instances = []

    for instance_type in instance_types:
        instances.append(instance_chooser(labels_set, instance_type))

    num_cases = len(instances)

    # This generates some dummy data data for the graph. Replace this with your data
    data = np.zeros((len(epoch_labels), len(instances)))

    # Set ground truth
    data[0] = [instance['value'] for instance in instances]

    for i, predictions in enumerate(prediction_set):
        for j, instance_predict in enumerate(instances):
            data[i + 1, j] = predictions[instance_predict['index']]

    # # Generates dimensional metrics for the graph
    ind = np.arange(num_cases)
    width = 1 / (len(epoch_labels)) / 1.25

    cols = ['black']

    cols.extend(c)

    # Iterates through the data and plots the bars
    for i, (epoch, col) in enumerate(zip(data, cols)):
        pos = ind + (i * width)
        ax4.bar(pos, epoch, width, label=epoch_labels[i], color=col)

    # ax4.set(xlabel='Ground Truth', ylabel='Predictions')
    ax4.set_xticks([])
    ax4.set_xlabel("Min Case        Median Case             Max Case")
    # plt.legend(loc='best')

    # Draw circles on the correlation plot corresponding
    for i in range(len(instances)):
        ground_truth = data[0][i]
        prediction = data[-1][i]

        ax2.add_patch(plt.Circle((ground_truth, prediction), 0.02, color='black', fill=False))

    fig.suptitle(fig_title)
    #fig.tight_layout()
    plt.savefig(fig_title)
    plt.close()


class TrainingProgress(Callback):

    def __init__(self, experiment, iteration, plot_back=5):
        super().__init__()

        self.file_name = experiment.trial.trial_name + "/" + experiment.name + "/" + \
                         "iteration" + str(iteration) + "_"
                         
        # These variables will store the best validation scores
        # Initialise the loss counters to infinity as this score will be beaten by any loss on the first epoch
        self.best_validation_losses = [np.inf]
        self.training_losses = []

        self.best_result_epochs = []

        # The number of best epochs back that need to be printed
        if plot_back > len(colours):
            plot_back = len(colours)
        self.plot_back = plot_back

        self.features = experiment.features
        self.labels = experiment.labels

        self.training_predictions = []
        self.validation_predictions = []

    def on_epoch_end(self, epoch, logs=None):

        loss_training_current = logs.get('loss')
        loss_validation_current = logs.get('val_loss')

        previous_best_validation_loss = self.best_validation_losses[-1]

        # If the validation loss beats the current best, append it to the list
        if loss_validation_current <= previous_best_validation_loss:

            # Update predictions
            self.training_predictions.append(self.model.predict(self.features.training_set()))
            self.validation_predictions.append(self.model.predict(self.features.validation_set()))

            message = "\n----\n Best validation score updated. \n New best: {val_score:.5f}. " \
                      "\n This represents an improvement of {percent:.1f}% improvement over the " \
                      "previous best ({previous:.5}). \n----\n"

            # Just remove the infinity if it's the first epoch
            if previous_best_validation_loss == np.inf:
                self.best_validation_losses.pop()

            else:

                percent = ((previous_best_validation_loss - loss_validation_current) / previous_best_validation_loss) \
                          * 100.0

                print(message.format(val_score=loss_validation_current, percent=percent,
                                     previous=previous_best_validation_loss))

            self.best_validation_losses.append(loss_validation_current)
            self.training_losses.append(loss_training_current)
            self.best_result_epochs.append(epoch + 1)

            num_results = len(self.best_result_epochs)

            if num_results > 1:
                lower_bound = max(0, (num_results - self.plot_back))

                x = self.best_result_epochs[lower_bound:]
                y = self.best_validation_losses[lower_bound:]

                plot_name = self.file_name + "Validation_Dashboard"
                plotter_foursquare(x, y, self.labels.validation_set(), self.validation_predictions[lower_bound:],
                                   plot_name)

                y = self.training_losses[lower_bound:]

                plot_name = self.file_name + "Training_Dashboard"
                plotter_foursquare(x, y, self.labels.training_set(), self.training_predictions[lower_bound:],
                                   plot_name)


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 250
    decay_step_2 = 300

    if epoch == decay_step or epoch == decay_step_2:
        return lr * decay_rate

    return lr
