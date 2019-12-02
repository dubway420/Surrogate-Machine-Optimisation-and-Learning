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

import pickle

# path = '/home/huw/PycharmProjects/Results/position_identification'
# base = "intact_core_rb"
#
# cases = tls.directories_in_path(path)


# case = path + cases[0] + '/' + cases[0]
# instance1 = ci(case)
#

# X = []
# Y = []
#
# for base in cases:
# case = path + '/' + base
# #
# instance1 = ci(case)
# x_coord_fuel, y_coord_fuel, z_coord_fuel = instance1.get_fuel_channel_xyz_positions()
# x_coord_inter, y_coord_inter, z_coord_inter = instance1.get_interstitial_channel_xyz_positions()
#
# print(len(x_coord_inter))


# print('\n===\n')
# print(x_coord_inter)
# fig = plt.figure()
# ax = Axes3D(fig)
# #
# ax.scatter(x_coord_fuel, y_coord_fuel, z_coord_fuel, c='b', marker='o', label="Fuel Channels")
# ax.scatter(x_coord_inter, y_coord_inter, z_coord_inter, c='r', marker='o', label="Interstitial Channels")
# ax.view_init(elev=10., azim=30)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.005), shadow=True, ncol=2)
# plt.show()

# plt.scatter(x_coord_fuel, y_coord_fuel, c='b', marker='o', label="Fuel Channels")
# plt.scatter(x_coord_inter, y_coord_inter, c='r', marker='o', label="Interstitial Channels")
#
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
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
regressors = [LinearRegression(), DecisionTreeRegressor(), Ridge()]
#
# cross_val_results = []
mean_squared_errors = []


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
# #
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
# #     cvr = cross_val_predict(regressor, X, Y, cv=8)
# #     cross_val_results.append(cvr)
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
