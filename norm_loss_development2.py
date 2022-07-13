from machine_learning.dataset_generators import DatasetSingleFrame, Displacements
import numpy as np
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
from scipy.stats import norm 

def adjustment_factor(i, modal_bin, std, norm_max, square=True):

    adjustment_factor = norm.pdf(i, modal_bin, std)/norm_max
    
    if square:
        adjustment_factor = adjustment_factor**2
    

    return adjustment_factor + 1


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

# =============================================================================

vals = np.linspace(0, 1, 100)
y = []

for val in vals:
    y.append(adjustment_factor(val, modal_bin, std, norm_max))


plt.scatter(vals, y, c='r')



plt.show()



