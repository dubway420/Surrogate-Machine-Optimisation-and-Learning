from core_parse import CoreInstance as ci
import tools as tls
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import itertools
import sys

import pickle


def cases_list(path_string):
    """ For a given directory, returns a list of instance cases, including the original path """

    cases = tls.directories_in_path(path_string)

    case_list = []

    for base in cases:
        case_list.append(path_string + base + '/' + base)

    return case_list


def features_and_labels(path_string, time=50, result="1"):
    """ Gets the features and labels from the folder of results"""

    cases = cases_list(path_string)

    X, Y = [], []

    for case in cases:
        instance = ci(case)

        X.append(instance.linear_crack_array_1d())
        Y.append(instance.get_result_at_time(time, result_columns=str(result)))

    return X, Y


def result_time_history(case, result="1", time_steps=271):
    """Gets a result time history for one case i.e. one result at each time"""

    instance = ci(case)

    time_history = []

    for time in range(1, time_steps + 1):
        time_history.append(instance.get_result_at_time(time, result_columns=str(result)))

    return time_history


with open('data.pkl', 'rb') as f:
    X, Y_50_1, Y_50_2, x_coord_inter_channel, y_coord_inter_channel = pickle.load(
        f)

case_number = 50

path_cases = '/media/huw/Seagate Expansion Drive/parmec_results/'

cases = cases_list(path_cases)

case = cases[case_number]

case_name = case.split('/')[-1]

regressor = LinearRegression()

X = np.where(np.array(X) > 1, 1, np.array(X))

cvr_1 = cross_val_predict(regressor, X, Y_50_1, cv=8)
cvr_2 = cross_val_predict(regressor, X, Y_50_2, cv=8)

axes = plt.gca()
axes.set_xlim([min(Y_50_1[case_number])*0.9, max(Y_50_1[case_number])*1.1])
axes.set_ylim([min(cvr_1[case_number])*0.9, max(cvr_1[case_number])*1.1])

plt.title(case_name)
plt.xlabel('Actual')
plt.ylabel('Prediction')

plt.scatter(Y_50_1[case_number], cvr_1[case_number], marker='o')
plt.plot(Y_50_1[case_number], Y_50_1[case_number])
plt.show()



#
# instance1 = ci(case)
#
# counts_local, counts_adjacent, counts_outer = [], [], []
#
# for i in range(1, instance1.last_channel(channel_type='inter') + 1):
#     local, adjacent, outer = instance1.get_cracks_per_layer(str(i), array_type='pos', channel_type='inter',
#                                                             inclusive=True)
#
#     counts_local.append(local)
#     counts_adjacent.append(adjacent)
#     counts_outer.append(outer)
#
# Y_time = result_time_history(case)
#
# for i, time_result in enumerate(Y_time):
#
#     plt.subplot(211)
#
#     title = "Instance: " + case_name + " - Concentration of Cracks "
#     plt.title(title)
#
#     plt.scatter(x_coord_inter_channel, y_coord_inter_channel, c=counts_local, marker='o', label="Local")
#
#     plt.axis('off')
#
#     plt.subplot(212)
#
#     plt.scatter(x_coord_inter_channel, y_coord_inter_channel, c=time_result, marker='o',
#             label="Magnitude of Displacement in x-direction (b = 1)")
#
#     title = "Instance: " + case_name + " - Result at timestep: " + str(i)
#
#     plt.title(title)
#     # plt.xlabel('x (meters)')
#     # plt.ylabel('y (meters)')
#     # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), shadow=True, ncol=2)
#
#     name = "./Figures1/" + case_name + "_i" + str(i) + '_r' + '1' + '.png'
#
#     plt.savefig(name)
    # plt.show()




# case_intact = '/home/huw/PycharmProjects/Results/position_identification/intact_core_rb'
#
# instance1 = ci(case_intact)
#
# x_coord_fuel, y_coord_fuel, z_coord_fuel = instance1.get_fuel_channel_xyz_positions()
# x_coord_inter, y_coord_inter, z_coord_inter = instance1.get_interstitial_channel_xyz_positions()


# counts_local, counts_adjacent, counts_outer = [], [], []
#
# for i in range(1, instance1.last_channel(channel_type='inter') + 1):
#     local, adjacent, outer = instance1.get_cracks_per_layer(str(i), array_type='pos', channel_type='inter',
#                                                             inclusive=True)
#
#     counts_local.append(local)
#     counts_adjacent.append(adjacent)
#     counts_outer.append(outer)

# j = 0
#
# i = 0
# x_coord_inter_channel, y_coord_inter_channel = [], []
# for x, y in zip(x_coord_inter, y_coord_inter):
#     i += 1
#     if i % 13 == 0:
#         x_coord_inter_channel.append(x)
#         y_coord_inter_channel.append(y)


# X, Y_50_1 = features_and_labels(path_cases)
# X, Y_50_2 = features_and_labels(path_cases, result="2")

# with open('data.pkl', 'wb') as f:
#     pickle.dump([X, Y_50_1, Y_50_2, x_coord_inter_channel, y_coord_inter_channel], f)

# plt.scatter(x_coord_inter_channel, y_coord_inter_channel, c='black', marker='o',
#             label="Interstitial Channels")

# plt.scatter(x_coord_inter_channel, y_coord_inter_channel, c=counts_local, marker='o', label="Local")
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
# plt.show()
#
# plt.scatter(x_coord_inter_channel, y_coord_inter_channel, c=Y_50_1[case_number], marker='o', label="Result")
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
# plt.show()
# print(x_coord_inter)
# fig = plt.figure()
# ax = Axes3D(fig)
# #
# ax.scatter(x_coord_fuel, y_coord_fuel, z_coord_fuel, c='b', marker='o', label="Fuel Channels")
# ax.scatter(x_coord_inter, y_coord_inter, z_coord_inter, c='r', marker='o', label="Interstitial Channels")
# ax.view_init(elev=10., azim=30)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), shadow=True, ncol=2)
# plt.show()


#
# plt.scatter(x_coord_fuel, z_coord_fuel, c='b', marker='o', label="Fuel Channels")
# plt.scatter(x_coord_inter, z_coord_inter, c='r', marker='o', label="Interstitial Channels")
#
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
# plt.show()


#     X.append(instance1.linear_crack_array_1d())
#     Y.append(instance1.get_result_at_time(50, result_columns="1"))

# with open('objs.pkl', 'wb') as f:
#     pickle.dump([X, Y, x_coord, y_coord], f)

# # # Getting back the objects:
# with open('objs.pkl', 'rb') as f:
#     X, Y, x_coord, y_coord = pickle.load(f)

# Y = np.array(Y) * 1000
# regressors = [LinearRegression(), DecisionTreeRegressor(), Ridge()]
#
# cross_val_results = []
# mean_squared_errors = []

# with open('objs.pkl', 'wb') as f:
#     pickle.dump([X, Y, x_coord, y_coord, cross_val_results], f)

# # Getting back the objects:
# with open('objs.pkl', 'rb') as f:
#     X, Y, x_coord, y_coord, cross_val_results = pickle.load(f)
#
# plt.scatter(x_coord, y_coord, c=Y[0])
#
# # plt.scatter(x_coord_fuel, y_coord_fuel, c='orange', label="Fuel Channels")
# # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
# plt.show()
#
# # X = np.where(np.array(X) > 1, 1, np.array(X))
# #cvr = cross_val_predict(regressor, X, Y, cv=8)
# # #     cross_val_results.append(cvr)
# # train_results = []
# #
# # cross_val_results = []
# #
# # for regressor in regressors:
# #
# #     regressor.fit(X, Y)
# # train_results.append(regressor.predict(X))
#
# #     print('\n===\n', regressor, '\n')
# #
# #
# #
# #     mse = mean_squared_error(Y, cvr)
# #
# #     print("Mean squared error:", mse)
# # #
# # k = 0
# # #
# # for i, channel_ground_result in enumerate(Y):
# #
# #     print('\n===\ncase:', i, '\n===\n')
# #     case_prediction = train_results[0][i]
# # #
# # #     # for ground, predict in zip(channel_ground_result, case_prediction):
# # #     #     print("Ground:", ground, "Prediction:", predict)
# # #
# #     k += 1
# #     if k == 10:
# #         fig, axs = plt.subplots(2)
# #
# #         mse = mean_squared_error(channel_ground_result, case_prediction)
# #
# #         # title = 'Case ' + str(i) + " - Mean Squared Error: " + str(round(mse, 3))
# #         # plt.title(title)
# #         axs[0].scatter(x_coord, y_coord, c=channel_ground_result)
# #
# #         axs[0].set_title("Parmec Ground Labels")
# #
# #         # for j in range(len(regressors)):
# #
# #         axs[1].scatter(x_coord, y_coord, c=case_prediction)
# #
# #         title = "Linear Regression - Training Results (MSE: " + str(round(mse, 3)) + ")"
# #         axs[1].set_title(title)
# #
# #         plt.show()
# #         k = 0
#
# # plt.scatter(x_coord, y_coord, c=Y[len(Y) - 1])
# # plt.show()
# #
# # print(len(test_predictions[0]))
# #
# # plt.scatter(x_coord, y_coord, c=test_predictions[0])
# # plt.show()
# # #
# # for ground, prediction in zip(Y[len(Y) - 1], test_predictions[0]):
# #     print("Ground:", ground, "Prediction:", prediction)
# # case = 'C:/Users/Huw/PycharmProjects/Results/batch11/batch11_12395_P40/batch11_12395_P40'
#
#
# # instance1 = CoreInstance(crack_file)
# #
# # # print(instance1.linear_crack_array_1d(levels="bottom"))
# # array = instance1.linear_crack_array_1d(levels="all")
# #
# # print(array)
