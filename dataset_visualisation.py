from machine_learning.dataset_generators import Cracks1D, Cracks2D, Cracks3D
from machine_learning.dataset_generators import DatasetSingleFrame, Displacements as Labels
from parmec_analysis.reactor_case import Parse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing as pre


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

print("\n == \n")
for line in dataset.summary():
    print(line)
print("\n == \n")

# ************************************************************************
# ************************************************************************
# FEATURES - This section shows three alternative encodings for the features
# ************************************************************************
# ************************************************************************

# =============================
# FEATURES 1-dimensional
#
# This format is suitable for a neural network with a dense input layer or shallow method
# =============================

features1D = Cracks1D(dataset)

print("\n == \n")
for line in features1D.summary():
    print(line)
print("\n == \n")

# =============================
# FEATURES 2-dimensional 
#
# This format is suitable for a neural network with a CNN2D input layer
# =============================

features2D = Cracks2D(dataset, extra_dimension=True)  # The extra dimension is added to make it suitable for CNNs

print("\n == \n")
for line in features2D.summary():
    print(line)
print("\n == \n")

# =============================
# FEATURES 3-dimensional 
#
# This format is suitable for a neural network with a CNN2D input layer
# =============================

features3D = Cracks3D(dataset)

print("\n == \n")
for line in features3D.summary():
    print(line)
print("\n == \n")

intact = Parse("intact_core_rb")

brick_coord_list_fuel = intact.get_brick_xyz_positions('xyz', channel_type="fuel", channels_only=0)
X = features1D.values[0]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.title.set_text("Try dragging this image around")
ax.scatter(brick_coord_list_fuel[0], brick_coord_list_fuel[1], brick_coord_list_fuel[2], c=X, cmap='gnuplot')
plt.legend()
ax.view_init(45, 45)
plt.show()

# ************************************************************************
# ************************************************************************
# LABELS - This section shows how to generate labels

# Ensure the file Y_dataset_C159_160_L11_12_T48_R1_all_T.npy is
# present in the same directory as this .py file
# ************************************************************************
# ************************************************************************

labels = Labels(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
                levels=levels_labels)

print("\n == \n")
for line in labels.summary():
    print(line)
print("\n == \n")

# This normalises the labels to the range 0 - 1

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

labels.transform(min_max_scaler)

sns.histplot(labels.values)
plt.show()
