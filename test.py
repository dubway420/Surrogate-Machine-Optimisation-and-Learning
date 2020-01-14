# Convo based on tutorial found at:
# https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca

# TODO WHAT IF MNIST IS CONVERTED TO BINARY? I.E. RATHER THAN 0 - 255, WE HAVE 0 OR 1? ROUND TO 0 OR 1?
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras import layers
from keras import models
from keras.activations import relu

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.summary()
