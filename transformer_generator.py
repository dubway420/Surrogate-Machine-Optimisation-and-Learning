from tensorflow.keras import models
from machine_learning.dataset_generators import Cracks, DatasetSingleFrame, Cracks3D, Displacements
from machine_learning.callbacks import correlation_foursquare, histogram_foursquare
from machine_learning.utils import find_nearest
from tensorflow.keras.losses import mean_squared_error as mse
import numpy as np
from sklearn import preprocessing as pre
import warnings
from os import listdir, mkdir
from parmec_analysis.utils import is_in
import sys
import pickle

dataset = DatasetSingleFrame()

dataset.augment(flip=1, rotate=(1, 2, 3))

Cracks = Cracks3D(dataset, array_type="Positions", levels="5-7")

labels = Displacements(dataset, channels="160", result_time=48, result_type="all", levels="12")

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

labels.transform(min_max_scaler)

print(np.min(labels.values))
print(np.max(labels.values))
print(np.mean(labels.values))
