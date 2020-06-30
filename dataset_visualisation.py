from machine_learning.dataset_generators import DatasetSingleFrame, Labels
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from machine_learning.dataset_generators import FeaturesConcentration1D as Features

# features = Features(dataset=DatasetSingleFrame("test", number_of_cases="all"))

# sns.set(style="darkgrid")
# sns.distplot(features.values)
# plt.show()


dataset_path = '~/localscratch/'

# Features
channels_features = 'all'
levels_features = 'all'
array_type = 'Positions Only'

# Labels
channels_labels = "all"
levels_labels = 'all'
result_type = 'all'
result_time = 48
result_column = 1

no_instances = 'all'
#
dataset = DatasetSingleFrame(dataset_path, number_of_cases=no_instances)

labels = Labels(dataset, channels=channels_labels, result_time=result_time, result_type=result_type, unit='millimeters',
                flat=False)

level_arrays = labels.values

level_averages = np.array(np.mean(level_arrays, axis=1))

sns.set(style="darkgrid")

for i in range (1, 4):
    sns.distplot(level_averages[:, i], label=str(i))
plt.legend()
plt.show()
#
# data = {}
#
# for i in range(level_averages.shape[1]):
#     data[str(i)] = level_averages[:, i].tolist()
#
# df = pd.DataFrame(data, columns=data.keys())
# correlation_matrix = df.corr()
#
# mask = np.zeros_like(correlation_matrix.values)
# mask[np.triu_indices_from(mask)] = True
#
# sns.heatmap(correlation_matrix, mask=mask)
# plt.title("Average of Each Level - Correlation")
# plt.xlabel("Core Level (n)")
# plt.ylabel("Core Level (n)")
# plt.show()

# print(level_averages.shape)

# # y_pos = np.arange(len(level_averages))
# #
# # plt.rcdefaults()
# # fig, ax = plt.subplots()
# #
# # ax.barh(y_pos, level_averages, align='center',
# #         color='green', ecolor='black')
# # ax.set_yticks(y_pos)
# # ax.set_yticklabels(y_pos)
# # ax.invert_yaxis()  # labels read top-to-bottom
# # ax.set_xlabel('Performance')
# # ax.set_title('How fast do you want to go today?')
# #
# # plt.show()
#
# # label_array = np.array(np.mean(level_arrays, axis=2))
#
# # print(label_array_channels.shape)
#
# # level_whole_core_flat = np.reshape(level_arrays, (level_arrays.shape[0] * level_arrays.shape[1], level_arrays.shape[2]))
# # print(level_whole_core_flat.shape)
# # print(level_whole_core_flat.shape)
# #
# # fig7, ax7 = plt.subplots()
# # ax7.set_title('Distribution across each level of the core - whole core')
# # ax7.boxplot(level_whole_core_flat)
# #
# # plt.show()
# # #
# level_central_top = level_arrays[:, 160]
# normalized_X = preprocessing.normalize(level_central_top)
# # print(level_central.shape)
# # print(normalized_X)
# # print(np.amax(normalized_X))
# # level_13 = level_arrays[:, :, 12]
#
#
# # data = {}
# #
# # for i in range(label_array.shape[1]):
# #     data[str(i)] = label_array[:, i].tolist()
#
#
# # df = pd.DataFrame(data, columns=data.keys())
# # correlation_matrix = df.corr()
# #
# # mask = np.zeros_like(correlation_matrix.values)
# # mask[np.triu_indices_from(mask)] = True
# #
# # sns.heatmap(correlation_matrix, mask=mask)
# # plt.title("Channel Correlation")
# # plt.xlabel("Core Channel (n)")
# # plt.ylabel("Core Channel (n)")
# # plt.show()
#
#
# # print(level_central.shape)
# #
# # fig7, ax7 = plt.subplots()
# # ax7.set_title('Distribution across each level of the core - central channel')
# # ax7.boxplot(level_central)
# #
# # plt.show()
#
# # fig, axes = plt.subplots(2, figsize=(5, 10))
# #
# #
# # for i, ax in enumerate(axes):
# #     ax.scatter(level_whole_core_flat[:, 10], level_whole_core_flat[:, (i+11)], c='blue')
# #     # ax.xlabel("Core Level 0 Label")
# #     # ax.ylabel(("Core Level " + str(i) + " Label"))
# #
# # plt.tight_layout()
# # plt.show()
#
#
#
