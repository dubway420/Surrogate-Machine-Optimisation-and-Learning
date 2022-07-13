from machine_learning.dataset_generators import DatasetSingleFrame, Displacements
import numpy as np
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
from scipy.stats import norm 

dataset = DatasetSingleFrame()


labels = Displacements(dataset, channels="160", result_time=48, result_type="all", levels="12")

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

labels.transform(min_max_scaler)

mean = np.mean(labels.values)
median = np.median(labels.values)
std = np.std(labels.values)

bins = np.histogram_bin_edges(labels.values, bins='auto')
histo = np.histogram(labels.values, bins=bins)

#Histogram of the data
hist = plt.hist(labels.values, bins=bins, label="Data Distribution")


modal_bin = (histo[1][np.argmax(histo[0])] + histo[1][np.argmax(histo[0])-1])/2

x= np.linspace(0, 1, 100)
norm_distr = norm.pdf(x, modal_bin, std)

adj_factor = np.max(histo[0]/np.max(norm_distr))
norm_distr = norm_distr*adj_factor

# Regular Normal Distribution
plt.scatter(x, norm_distr, c='r', label='Regular Normal Distribution')

norm_distr_adj = []
norm_distr_adj2 = []
norm_max = np.max(norm_distr)


for i, val in zip(x, norm_distr):

    # adj = 1

    # if i < modal_bin:

    adj = (val/norm_max) ** 2
    adj2 = val/norm_max
    norm_distr_adj.append(adj*val)
    norm_distr_adj2.append(adj2*val)

plt.scatter(x, norm_distr_adj, c='y', label='Adjusted Normal Distribution')
plt.scatter(x, norm_distr_adj2, c='g', label='Non Square')    

plt.plot([modal_bin, modal_bin], [0, norm_max], '--')

plt.yticks([0, 400], [1, 2])
plt.legend(loc='upper right')

plt.show()
