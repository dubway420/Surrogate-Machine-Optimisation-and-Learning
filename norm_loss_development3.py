from machine_learning.losses import mse_norm_adjusted, huber_loss, huber_loss_mean
from machine_learning.dataset_generators import DatasetSingleFrame, Displacements
import numpy as np
from sklearn import preprocessing as pre
from scipy.stats import norm 
import tensorflow as tf





# Load the dataset and labels
dataset = DatasetSingleFrame()
labels = Displacements(dataset, channels="160", result_time=48, result_type="all", levels="12")

# Scale the data
min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))
labels.transform(min_max_scaler)

# =============================================================================
# Everything after this will need to be added to the trial_common_parameters.py file


# Calculate the mean, median and standard deviation of the data

std = np.std(labels.values)

bins = np.histogram_bin_edges(labels.values, bins='auto')
histo = np.histogram(labels.values, bins=bins)

modal_bin = (histo[1][np.argmax(histo[0])] + histo[1][np.argmax(histo[0])-1])/2

x= np.linspace(0, 1, 100)
norm_distr = norm.pdf(x, modal_bin, std)
norm_max = np.max(norm_distr)

loss = mse_norm_adjusted(modal_bin, std, norm_max, mean=False)

y_true = np.array([0.01, 0.21, 0.32, 0.49, 0.71, 0.99])
y_pred = np.array([0, 0.2, 0.31, 0.5, 0.7, 1])

print(loss(y_true, y_pred)*10000)
print(mse(y_true, y_pred))
print(huber_loss(y_true, y_pred))
# print(huber_loss_mean(y_true, y_pred))