from machine_learning.dataset_generators import Cracks1D
from machine_learning.dataset_generators import DatasetSingleFrame, Displacements as Labels
from parmec_analysis.reactor_case import Parse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing as pre
from time import sleep
import numpy as np
from sklearn import linear_model, svm, tree
from tensorflow.keras.losses import mean_squared_error as mse


# =============================
# THIS SECTION IS ALL THE PARAMETERS FOR THE FEATURES AND LABELS
# =============================

# Features
channels_features = 'all'
levels_features = 'all'
array_type = 'Positions Only'

# Labels
channels_labels = "160"
levels_labels = '12'
result_type = 'all'
result_time = 48
result_column = 1

no_instances = 'all'

# =============================
# LOAD THE DATASET - Ensure the file dataset_cases.pkl is present in the same directory as this .py file
# =============================

dataset = DatasetSingleFrame()

# print("\n == \n")
# for line in dataset.summary():
#     print(line)
# print("\n == \n")

# =============================
# FEATURES 1-dimensional
#
# This format is suitable for a neural network with a dense input layer or shallow method
# =============================

features1D = Cracks1D(dataset)

# print("\n == \n")
# for line in features1D.summary():
#     print(line)
# print("\n == \n")

# ************************************************************************
# ************************************************************************
# LABELS - This section shows how to generate labels

# Ensure the file Y_dataset_C159_160_L11_12_T48_R1_all_T.npy is
# present in the same directory as this .py file
# ************************************************************************
# ************************************************************************

labels = Labels(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
                levels=levels_labels)

# print("\n == \n")
# for line in labels.summary():
#     print(line)
# print("\n == \n")

# This normalises the labels to the range 0 - 1

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

labels.transform(min_max_scaler)

# ===================================================================================================

# =============================
# Test set
# =============================

test_set = DatasetSingleFrame(name="test_set")

# print("\n == \n")
# for line in test_set.summary():
#     print(line)
# print("\n == \n")

# =============================
# FEATURES 1-dimensional
#
# This format is suitable for a neural network with a dense input layer or shallow method
# =============================

features1D_test = Cracks1D(test_set)

# print("\n == \n")
# for line in features1D_test.summary():
#     print(line)
# print("\n == \n")

# ************************************************************************
# ************************************************************************
# LABELS - This section shows how to generate labels

# Ensure the file Y_dataset_C159_160_L11_12_T48_R1_all_T.npy is
# present in the same directory as this .py file
# ************************************************************************
# ************************************************************************

labels_test = Labels(test_set, channels=channels_labels, result_time=result_time, result_type=result_type,
                levels=levels_labels)

# print("\n == \n")
# for line in labels_test.summary():
#     print(line)
# print("\n == \n")

# This normalises the labels to the range 0 - 1

labels_test.transform(min_max_scaler)



# =========================================================================
# Model Training
# =========================================================================

training_features = features1D.training_set()
training_labels = labels.training_set()

test_features = features1D_test.values
test_labels = labels_test.values



# Correlation square
# histogram

models = [
          linear_model.LinearRegression(), 
          linear_model.HuberRegressor(), 
          svm.SVR(), 
          tree.DecisionTreeRegressor()]

model_names = [
               "Linear regression",
               "Huber regression",
               "Support vector regression",
               "Decision Tree Regression"]

for model, name in zip(models, model_names):

    print(f"\n========= {name} =========\n")

    trained_model = model.fit(training_features, training_labels.ravel())

    training_predictions = trained_model.predict(training_features)

    training_loss = np.mean(mse(training_predictions, training_labels))

    print("Training loss: {:e}".format(training_loss))

    test_predictions = trained_model.predict(test_features)

    test_loss = np.mean(mse(test_predictions, test_labels))

    print("Test loss: {:.2e}".format(test_loss))

    print("\n")


