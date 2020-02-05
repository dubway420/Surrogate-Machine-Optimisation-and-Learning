from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from keras.models import Model
import math as m
import numpy as np

# TODO a 2 column plot for each filter - the filter itself in left column, the feature map on right column

epochs = 10

# download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# create model
conv2d = Sequential()
conv2d.add(Conv2D(64, kernel_size=3, activation='sigmoid', input_shape=(28, 28, 1)))
conv2d.add(Conv2D(32, kernel_size=3, activation='relu'))
conv2d.add(Flatten())
conv2d.add(Dense(10, activation='softmax'))

conv2d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

conv2d.summary()

for layer in conv2d.layers:
    print(layer.name)

model = Model(inputs=conv2d.inputs, outputs=conv2d.layers[0].output)

# conv2d.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)

# conv2d.save_weights("conv2d.h5")
conv2d.load_weights("conv2d.h5")

case = 50
channel = 15

plt.imshow(X_train[case].reshape(28, 28), cmap='gray')
plt.show()

feature_map = model.predict(
    X_train[case].reshape(1, X_train[case].shape[0], X_train[case].shape[1], X_train[case].shape[2]))

plt.imshow(feature_map[0, :, :, channel], cmap='gray')
plt.show()

filters, biases = conv2d.layers[0].get_weights()
filter = filters[:, :, :, channel].reshape(filters.shape[0], filters.shape[1])

plt.imshow(filter, cmap='gray')
plt.show()

# filter = filters[:, :, :, 1].reshape(filters.shape[0], filters.shape[1])
#
# plt.imshow(filter, cmap='gray')
# plt.show()

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# # plot first few filters
n_filters, rows_cols = filters.shape[-1], int(m.sqrt(filters.shape[-1]))

fig, axs = plt.subplots(rows_cols, rows_cols)

for row_no in range(rows_cols):

    # plot each channel separately
    for col_no in range(rows_cols):
        filter_no = (row_no * rows_cols) + col_no
        filter = filters[:, :, :, filter_no].reshape(filters.shape[0], filters.shape[1])

        filter = np.where(filter > 0.5, 1, filter)
        filter_Shreshold = np.where(filter <= 0.5, 0, filter)

        # specify subplot and turn of axis
        # ax = plt.subplot(n_filters, 3, ix)
        ax = axs[row_no, col_no]
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        ax.imshow(filter_Shreshold, cmap='gray')

# show the figure
plt.show()
