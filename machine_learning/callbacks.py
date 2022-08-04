import matplotlib as mpl

# Agg backend will render without X server on a compute node in batch
#mpl.use('Agg')
import numpy as np
from matplotlib import mlab
# from sympy.stats import density
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from parmec_analysis.utils import plot_names_title, is_in
from parmec_analysis.visualisation import CoreView, TrainingHistoryRealTime, autolabel
from tensorflow.keras.losses import mean_squared_error as mse, mean_absolute_error as mae
from tensorflow.keras.losses import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score
import tensorflow as tf

# from keras.callbacks import ModelCheckpoint

#tf.enable_eager_execution()

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

    ax2.plot([0.1, 1.1], '--', c='black', alpha=0.5, label='+/- 10%')
    ax2.plot([0, 0.9], '--', c='black', alpha=0.5)

    # Draw a straight line from 0 to 1 denoting a 'perfect' result
    ax2.plot([0.2, 1.2], 'k:', c='black', alpha=0.5, label='+/- 20%')
    ax2.plot([0, 1], c='black')
    ax2.plot([0, 0.8], 'k:', c='black', alpha=0.5)

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
    # fig.tight_layout()
    plt.savefig(fig_title)
    # plt.show()
    plt.close()


def correlation_foursquare(x, y, labels_set, prediction_set, fig_title, binary_delineaters=(0.2, 0.37, 0.6), y_lim=True):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # Draw a straight line from 0 to 1 denoting a 'perfect' result
    ax1.plot([0.1, 1.1], '--',  alpha=0.5, label='+/- 10%')
    ax1.plot([-0.1, 0.9], '--',  alpha=0.5)

    # Draw a straight line from 0 to 1 denoting a 'perfect' result
    ax1.plot([0.2, 1.2], 'k:',  alpha=0.5, label='+/- 20%')
    ax1.plot([0, 1], c='black')
    ax1.plot([-0.2, 0.8], 'k:',  alpha=0.5)
    

    epoch_labels = ["Ground Truth"]

    predictions = prediction_set[-1]
    epoch = "Epoch " + str(x[-1])

    set_LT10 = [[], []]  # within 10%
    set_LT20 = [[], []]  # within 20%
    set_20P = [[], []]  # outside 20%

    set20P_indices = []  # indices of the outlier cases

    # binary_delineater = np.median(labels_set)

    for i, (label, prediction) in enumerate(zip(labels_set, predictions)):

        # =======================
        # Stuff for the first graph
        # =======================

        difference = abs(label - prediction)

        if difference <= 0.1:
            set_LT10[0].append(label)
            set_LT10[1].append(prediction)

        elif difference <= 0.2:
            set_LT20[0].append(label)
            set_LT20[1].append(prediction)

        else:
            set_20P[0].append(label)
            set_20P[1].append(prediction)
            set20P_indices.append(i)

    number_of_predictions = len(predictions)

    LT10_percentage = str(len(set_LT10[0])) + " (" + str(
        round(len(set_LT10[0]) / number_of_predictions * 100, 1)) + "%)"
    LT20_percentage = str(len(set_LT20[0])) + " (" + str(
        round(len(set_LT20[0]) / number_of_predictions * 100, 1)) + "%)"
    P20_percentage = str(len(set_20P[0])) + " (" + str(round(len(set_20P[0]) / number_of_predictions * 100, 1)) + "%)"

    ax1.scatter(set_LT10[0], set_LT10[1], label=LT10_percentage, c='purple')
    ax1.scatter(set_LT20[0], set_LT20[1], label=LT20_percentage, c='grey')
    ax1.scatter(set_20P[0], set_20P[1], label=P20_percentage, c='y')

    m, b = np.polyfit(labels_set.flatten(), predictions, 1)
    #
    ax1.plot(labels_set, m * labels_set + b)

    ax1.legend()
    ax1.set(xlabel='Ground Truth', ylabel='Predictions')

    line_equation = "y = " + str(m) + "x + " + str(b)
    ax1.title.set_text(line_equation)
    ax2.title.set_text(epoch)

    correlation_binary_split(ax2, binary_delineaters[1], labels_set, predictions)

    correlation_binary_split(ax3, binary_delineaters[0], labels_set, predictions)

    losses = "MSE: " + str(np.mean(mse(labels_set, predictions).numpy()).round(4)) + " MAE: " + str(
        np.mean(mae(labels_set, predictions).numpy()).round(4)) + " MAPE: " + str(
        np.mean(mape(labels_set, predictions).numpy()).round(2))

    # print(losses)
    ax3.title.set_text(losses)

    correlation_binary_split(ax4, binary_delineaters[2], labels_set, predictions)
    title_four = "$R^2$ " + str(r2_score(labels_set, predictions))
    ax4.title.set_text(title_four)

    # fig.title("Testing")
    fig.suptitle(fig_title)
    # fig.tight_layout()
    
    if y_lim:
        
        ax1.set_ylim(0, 1.0)
        ax2.set_ylim(0, 1.0)
        ax3.set_ylim(0, 1.0)
        ax4.set_ylim(0, 1.0)

    save_name = fig_title + "_correlation"
    plt.savefig(save_name)
    # plt.show()
    plt.close()

    outlier_title = fig_title + "_outliers.csv"
    # print(np.array(set20P_indices))
    np.savetxt(outlier_title, np.array(set20P_indices).astype(int), delimiter=',', fmt="%i")


def histogram_foursquare(x, y, labels_set, prediction_set, fig_title, binary_delineaters=(0.2, 0.37, 0.6)):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    predictions = prediction_set[-1]

    epoch = "Epoch " + str(x[-1])

    # =============================================
    # TOP LEFT IMAGE
    # =============================================

    n, bins, patches = ax1.hist(labels_set.flatten(), color='grey', label="Ground Truth",
                                density=True)

    graph_height = max(n)

    y = histo_fit(labels_set.flatten(), bins)

    ax1.plot(bins, y, '--', c='black')

    n, bins, patches = ax1.hist(predictions, color='green', label="Predictions", density=True, alpha=0.5)

    y = histo_fit(predictions, bins)

    gt_mean = np.mean(labels_set.flatten())
    pred_mean = np.mean(predictions)

    diff_mean = gt_mean - pred_mean

    ax1.plot([gt_mean, gt_mean], [0, graph_height], color='black', label='Mean (GT)')
    ax1.plot([pred_mean, pred_mean], [0, graph_height], color='green', label='Mean (Pred.)')

    gt_std = np.std(labels_set.flatten())
    pred_std = np.std(predictions)

    ax1.plot([(gt_mean - gt_std), (gt_mean - gt_std)], [0, graph_height], '--', color='black', label='STD. (GT)')
    ax1.plot([(gt_mean + gt_std), (gt_mean + gt_std)], [0, graph_height], '--', color='black', label='STD. (Pred.)')

    ax1.plot([(pred_mean - pred_std), (pred_mean - pred_std)], [0, graph_height], '--', color='green')
    ax1.plot([(pred_mean + pred_std), (pred_mean + pred_std)], [0, graph_height], '--', color='green')

    mean_label = "Ground Truth Mean: " + str(round(gt_mean, 1)) + ", Predictions Mean: " + str(
        round(pred_mean, 1)) + " (diff. " + str(round(diff_mean, 1)) + ")"

    ax1.title.set_text(mean_label)

    ax1.legend()
    ax1.plot(bins, y, '--', c='green')

    # =============================================
    # TOP RIGHT IMAGE
    # =============================================

    predictions_right = predictions[np.where(predictions > binary_delineaters[1])]
    labels_right = labels_set[np.where(labels_set > binary_delineaters[1])]

    n, bins, patches = ax2.hist(labels_right.flatten(), color='grey', label="Ground Truth",
                                density=True)

    y = histo_fit(labels_right.flatten(), bins)

    # ax2.plot(bins, y, '--', c='black')

    n, bins, patches = ax2.hist(predictions_right, color='green', label="Predictions", density=True, alpha=0.5)

    y = histo_fit(predictions_right, bins)

    diff_STD = gt_std - pred_std
    std_label = "Ground Truth STD: " + str(round(gt_std, 1)) + ", Predictions STD: " + str(
        round(pred_std, 1)) + " (diff. " + str(round(diff_STD, 1)) + ")"
    ax3.title.set_text(std_label)
    # ax2.plot(bins, y, '--', c='green')

    # =============================================
    # BOTTOM LEFT IMAGE
    # =============================================

    predictions_left = predictions[np.where(predictions <= binary_delineaters[1])]
    labels_left = labels_set[np.where(labels_set <= binary_delineaters[1])]

    n, bins, patches = ax3.hist(labels_left.flatten(), color='grey', label="Ground Truth",
                                density=True)

    y = histo_fit(labels_left.flatten(), bins)

    # ax3.plot(bins, y, '--', c='black')

    n, bins, patches = ax3.hist(predictions_left, color='green', label="Predictions", density=True, alpha=0.5)

    y = histo_fit(predictions_left, bins)

    # ax3.plot(bins, y, '--', c='green')

    # =============================================
    # BOTTOM RIGHT IMAGE
    # =============================================

    predictions_outliers = predictions[np.where(predictions > binary_delineaters[2])]
    labels_outliers = labels_set[np.where(labels_set > binary_delineaters[2])]

    n, bins, patches = ax4.hist(labels_outliers.flatten(), color='grey', label="Ground Truth",
                                density=True)

    y = histo_fit(labels_outliers.flatten(), bins)

    # ax4.plot(bins, y, '--', c='black')

    n, bins, patches = ax4.hist(predictions_outliers, color='green', label="Predictions", density=True, alpha=0.5)

    y = histo_fit(predictions_outliers, bins)

    # ax4.plot(bins, y, '--', c='green')

    fig.suptitle(fig_title)

    save_name = fig_title + "_histogram"
    plt.savefig(save_name)

    plt.close()


def correlation_binary_split(ax, binary_delineater, labels_set, predictions):
    true_negative = [[], []]
    false_negative = [[], []]
    negatives = 0

    true_positive = [[], []]
    false_positive = [[], []]
    positives = 0

    for label, prediction in zip(labels_set, predictions):

        # =======================
        # Stuff for the second graph
        # =======================

        if label <= binary_delineater:
            negatives += 1

            if prediction <= binary_delineater:

                true_negative[0].append(label)
                true_negative[1].append(prediction)

            else:

                false_positive[0].append(label)
                false_positive[1].append(prediction)

        else:
            positives += 1

            if prediction > binary_delineater:

                true_positive[0].append(label)
                true_positive[1].append(prediction)

            else:

                false_negative[0].append(label)
                false_negative[1].append(prediction)

    negatives = len(true_negative[0]) + len(false_positive[0])
    positives = len(true_positive[0]) + len(false_negative[0])

    ax.plot([binary_delineater, binary_delineater], '--', c='black', alpha=0.5)
    ax.plot([binary_delineater, binary_delineater], [0, 1], '--', c='black', alpha=0.5)

    TN_label = "True Negative (" + str(round(len(true_negative[0]) / negatives * 100, 1)) + "%)"
    ax.scatter(true_negative[0], true_negative[1], label=TN_label, c='blue')

    FP_label = "False Positive (" + str(round(len(false_positive[0]) / negatives * 100, 1)) + "%)"
    ax.scatter(false_positive[0], false_positive[1], label=FP_label, c='hotpink')

    ax.scatter([0, 1], [0, 1], alpha=1, label=" ", c="white")

    TP_label = "True Positive (" + str(round(len(true_positive[0]) / positives * 100, 1)) + "%)"
    ax.scatter(true_positive[0], true_positive[1], label=TP_label, c='green')

    FN_label = "False Negative (" + str(round(len(false_negative[0]) / positives * 100, 1)) + "%)"
    ax.scatter(false_negative[0], false_negative[1], label=FN_label, c='sandybrown')

    ax.set(xlabel='Ground Truth', ylabel='Predictions')

    ax.legend()


class TrainingProgress(Callback):

    def __init__(self, experiment, iteration, plot_back=5, save_model=False):
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

        self.epochs = experiment.trial.epochs

        self.save_model = save_model

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

                plot_name = self.file_name + "Validation"

                histogram_foursquare(x, y, self.labels.validation_set(), self.validation_predictions[lower_bound:],
                                     plot_name)

                correlation_foursquare(x, y, self.labels.validation_set(), self.validation_predictions[lower_bound:],
                                       plot_name)

                y = self.training_losses[lower_bound:]

                plot_name = self.file_name + "Training"
                histogram_foursquare(x, y, self.labels.training_set(), self.training_predictions[lower_bound:],
                                     plot_name)

                correlation_foursquare(x, y, self.labels.training_set(), self.training_predictions[lower_bound:],
                                       plot_name)

                if self.save_model:
                    save_name = self.file_name[:-1] + ".mod"
                    self.model.save(save_name)

        # Save the model at the half way point
        if epoch == int(self.epochs / 2) and self.save_model:

            save_name = self.file_name[:-1] + "_halfway.mod"
            self.model.save(save_name)

        # Save the model on the final epoch
        if epoch == self.epochs and self.save_model:

            save_name = self.file_name[:-1] + "_end.mod"
            self.model.save(save_name)


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 250
    decay_step_2 = 300

    if epoch == decay_step or epoch == decay_step_2:
        return lr * decay_rate

    return lr
