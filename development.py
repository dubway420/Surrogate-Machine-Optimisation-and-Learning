from core_parse import CoreInstance as ci
import tools as tls
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from pyimagesearch.models import create_mlp


def features_and_labels_single_frame(path_string, time=50, result="1", x_type='positions'):
    """ Gets the features and labels from the folder of results"""

    cases = tls.cases_list(path_string)

    X, Y = [], []

    for case in cases:
        instance = ci(case)

        X.append(instance.linear_crack_array_1d(array_type=x_type))
        Y.append(instance.get_result_at_time(time, result_columns=str(result)))

    return X, Y


##########
# CAN COMMENT ALL THIS OUT AFTER DOING IT ONCE
##########

# case_intact = 'C:/Users/Huw/PycharmProjects/Results/intact_core'
# instance_intact = ci(case_intact)
#
# # spatial coordinates of the interstitial channels - two lists, each of length 321.
# # first list contains x, second list contains y coordinates
# channel_coord_list_inter = instance_intact.get_brick_xyz_positions('xy', channel_type='inter')
#
# # Location of results
case_root = 'D:/parmec_results/'
#
# X, Y = features_and_labels_single_frame(case_root, time=48)
#
# with open('objs.pkl', 'wb') as f:
#     pickle.dump([X, Y, channel_coord_list_inter], f)

# ##########

# Load the dataset and channel coordinates
with open('objs.pkl', 'rb') as f:
    X, Y, channel_coord_list_inter, cross_val_results_traditional, mean_squared_errors = pickle.load(f)

#
# regressors_traditional = [LinearRegression, DecisionTreeRegressor, Ridge]
regressors_NN = [mlp]
#
# cross_val_results = []
# mean_squared_errors = []
# train_results = []
#
# for regressor in regressors:
#     print('\n=======\n')
#     print(regressor.__name__)
#
#     print("Performing cross-validation.")
#     cvr = cross_val_predict(regressor(), X, Y, cv=8)
#     cross_val_results.append(cvr)
#
#     mse = mean_squared_error(Y, cvr)
#     mean_squared_errors.append(mse)
#
#     print("Cross validation mean squared error: ", mse)

# regressor.fit(X, Y)

# with open('objs.pkl', 'wb') as f:
#     pickle.dump([X, Y, channel_coord_list_inter, cross_val_results, mean_squared_errors], f)


######################################
# ############## Plotting ############
######################################

# cases = tls.cases_list(case_root)
#
# case_numbers = 6, 50, 25, 448
#
# fig, axs = plt.subplots((len(regressors) + 1), len(case_numbers), figsize=(10, 9))
#
# size = 15
#
# vals = []
# scatter = []
#
# for j in range(len(case_numbers)):
#
#     case_number = case_numbers[j]
#
#     scatter.append(axs[0, j].scatter(channel_coord_list_inter[0], channel_coord_list_inter[1], c=Y[case_number], s=size,
#                               cmap='nipy_spectral'))
#
#     axs[0, j].set_title(cases[case_numbers[j]].split('/')[-1])
#
#     if j == 0: axs[0, j].set_ylabel("Ground Truth Labels")
#
#     # append the minimum and maximum values
#     vals.append(np.amin(Y[case_number]))
#     vals.append(np.amin(Y[case_number]))
#
#     for i in range(len(regressors)):
#
#         regressor_name = regressors[i].__name__
#
#         result = cross_val_results[i]
#         scatter.append(
#             axs[i + 1, j].scatter(channel_coord_list_inter[0], channel_coord_list_inter[1], c=result[case_number],
#                                   s=size, cmap='nipy_spectral'))
#
#         if j == 0: axs[i + 1, j].set_ylabel(regressor_name)
#
#         vals.append(np.amin(result[case_number]))
#         vals.append(np.amax(result[case_number]))
#
# min_overall = np.amin(vals)
# max_overall = np.amax(vals)
#
# for ax in axs.flatten():
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#
# for s in scatter:
#     s.set_clim(min_overall, max_overall)
#
# plt.show()

######################################


# #cvr = cross_val_predict(regressor, X, Y, cv=8)
# # #     cross_val_results.append(cvr)
# # train_results = []
# #
# # cross_val_results = []
# #
# # for regressor in regressors:
# #
# #
# # train_results.append(regressor.predict(X))
#
# #     print('/n===/n', regressor, '/n')
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
# #     print('/n===/ncase:', i, '/n===/n')
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
