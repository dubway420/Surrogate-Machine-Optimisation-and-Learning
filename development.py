from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import Flatten
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers.core import Dense
from pyimagesearch import models
from keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt


#
# def features_and_labels_single_frame(path_string, time=50, result="1", x_type='positions'):
#     """ Gets the features and labels from the folder of results"""
#
#     cases = tls.cases_list(path_string)
#
#     X, Y = [], []
#
#     for case in cases:
#         instance = ci(case)
#
#         X.append(instance.linear_crack_array_1d(array_type=x_type))
#         Y.append(instance.get_result_at_time(time, result_columns=str(result)))
#
#     return X, Y


def multi_layer_perceptron(input_dims, output_dims):
    # define our MLP network
    model = Sequential()

    model.name = "Multi-layer Perceptron"

    model.add(Dense(8, input_dim=input_dims, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_dims, activation="linear"))

    # return our model
    return model


def wider_model(input_dims, output_dims):
    # create model
    model = Sequential()

    model.name = "Wider Perceptron"

    model.add(Dense(20, input_dim=input_dims, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(output_dims, kernel_initializer='normal'))

    return model


def cnn1D(input_dims, output_dims):
    # create model
    model = Sequential()  # add model layers
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_dims))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dims, activation='softmax'))

    # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def cnn2D(input_dims, output_dims):
    # create model
    model = Sequential()  # add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_dims))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dims, activation='softmax'))

    # Name the neural network
    model.name = "CNN 1"

    # model.compile(loss='mean_squared_error', optimizer='adam')
    return model


#

##########
# LOADS DATA FROM CASES - CAN COMMENT ALL THIS OUT AFTER DOING IT ONCE
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

# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# X_1D = numpyX.reshape((numpyX.shape[0], numpyX.shape[1], 1))
# X_2D = numpyX.reshape((numpyX.shape[0], 284, 7, 1))
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# #
# # # 1D data
# # # X_train_1D = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# # # X_test_1D = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# #
# # 2D data
# X_train_2D = X_train.reshape(X_train.shape[0], 284, 7, 1)
# X_test_2D = X_test.reshape(X_test.shape[0], 284, 7, 1)
# #
# regressors_traditional = [LinearRegression, DecisionTreeRegressor, Ridge]
#
# regressors_NN_1D = [
#                     multi_layer_perceptron(X_train.shape[1], len(Y_train[0])),
#                     wider_model(X_train.shape[1], len(Y_train[0]))
#                     ]
#
# regressors_NN_2D = [
#                     # cnn2D((X_train_2D.shape[1], X_train_2D.shape[2], X_train_2D.shape[3]), len(Y_train[0])),
#                     models.create_cnn(X_train_2D.shape[2], X_train_2D.shape[1], 1, regress=True)
#                     ]
#
# opt = Adam(lr=1e-3, decay=1e-3 / 200)
#
# model_names = []
#
# training_results = []
# testing_results = []
#
# training_accuracies = []
# testing_accuracies = []
#
# #################################################
# ########### Traditional Methods #################
# #################################################
#
# # print("\n Starting 1D input models \n")
#
#
# # for regressor in regressors_traditional:
# #
# #     print("\n=============\n")
# #
# #     model_name = regressor.__name__
# #     print(model_name)
# #     print("\n")
# #
# #     model = regressor()
# #     model.fit(X_train, Y_train)
# #     training_result = model.predict(X_train)
# #     testing_result = model.predict(X_test)
# #
# #     training_results.append(training_result)
# #     testing_results.append(testing_result)
# #
# #     mse_training = round(mean_squared_error(training_result, Y_train), 2)
# #     mse_testing = round(mean_squared_error(testing_result, Y_test), 2)
# #
# #     training_accuracies.append(mse_training)
# #     testing_accuracies.append(mse_testing)
# #
# #     print("Training accuracy: ", mse_training)
# #     print("Testing accuracy: ", mse_testing)
# #
#
# #################################################
# ########### Neural Networks #####################
# #################################################
# #
# # for model in regressors_NN_1D:
# #     model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
# #     model.fit(X_train, Y_train, epochs=500, validation_data=(X_test, Y_test), verbose=0)
# #
# #     print("\n=============\n")
# #
# #     model_name = model.name
# #     print(model_name)
# #     print("\n")
# #
# #     model_names.append(model_name)
# #
# #     training_result = model.predict(X_train)
# #     testing_result = model.predict(X_test)
# #
# #     training_results.append(training_result)
# #     testing_results.append(testing_result)
# #
# #     mse_training = round(mean_squared_error(training_result, Y_train), 2)
# #     mse_testing = round(mean_squared_error(testing_result, Y_test), 2)
# #
# #     training_accuracies.append(mse_training)
# #     testing_accuracies.append(mse_testing)
# #
# #     print("Training accuracy: ", mse_training)
# #     print("Testing accuracy: ", mse_testing)
# #
# #     print("\n=============\n")
#
#
# #################################################
# ################ Convo Nets #####################
# #################################################
#
# print("\n Starting 2D input models \n")
#
# for model in regressors_NN_2D:
#     model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
#
#     model.summary()
    # model.fit(X_train_2D, Y_train, epochs=100, validation_data=(X_test_2D, Y_test))
    #
    # print("\n=============\n")
    #
    # model_name = model.name
    # print(model_name)
    # print("\n")
    #
    # model_names.append(model_name)
    #
    # training_result = model.predict(X_train_2D)
    # testing_result = model.predict(X_test_2D)
    #
    # training_results.append(training_result)
    # testing_results.append(testing_result)
    #
    # mse_training = round(mean_squared_error(training_result, Y_train), 2)
    # mse_testing = round(mean_squared_error(testing_result, Y_test), 2)
    #
    # training_accuracies.append(mse_training)
    # testing_accuracies.append(mse_testing)
    #
    # print("Training accuracy: ", mse_training)
    # print("Testing accuracy: ", mse_testing)
    #
    # print("\n=============\n")
#
# names = ['LR', 'DT', 'R', 'MLP', 'WP', 'CNN 1', 'CNN 2']
#
# x = np.arange(len(names))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, training_accuracies, width, label='Training')
# rects2 = ax.bar(x + width/2, testing_accuracies, width, label='Testing')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Mean Squared Error')
# # ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(names)
# ax.legend()
#
# fig.tight_layout()
#
# plt.show()


# regressors_NN.append(models.create_cnn(7, 284, 1, regress=True))

#

#
# cross_val_results = []
# mean_squared_errors = []
# train_results = []
#
# for regressor in regressors_traditional:
#     print('\n=======\n')
#     print(regressor.__name__)
#
#     print("Performing cross-validation.")
#     cvr = cross_val_predict(regressor(), X, Y, cv=8)
#     cross_val_results.append(cvr)
#
#     mse = mean_squared_error(Y, cvr)
#     mean_squared_errors.append(mse)

# print("Cross validation mean squared error: ", mse)

# regressor.fit(X, Y)

# with open('objs.pkl', 'wb') as f:
#     pickle.dump([X, Y, channel_coord_list_inter, cross_val_results, mean_squared_errors], f)


######################################
# ############## Plotting ############
######################################
# #
# cases = tls.cases_list(case_root)
# regressors = regressors_traditional
#
# case_numbers = 121, 303, 278
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
#         # regressor_name = regressors[i].__name__
#         regressor_name = regressors[i].__name__
#         result = cross_val_results_traditional[i]
#         scatter.append(
#             axs[i + 1, j].scatter(channel_coord_list_inter[0], channel_coord_list_inter[1], c=result[case_number],
#                                   cmap='nipy_spectral', s=size))
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
# # for s in scatter:
# #     s.set_clim(min_overall, max_overall)
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

######################################
# ########### Histogram ##############
######################################

sns.set(color_codes=True)

sns.distplot(Y.flatten(), rug=True)
plt.ylabel("Frequency")
plt.xlabel("Displacement value (mm)")
plt.show()
