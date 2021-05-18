from machine_learning.callbacks import histogram
import numpy as np


x = np.random.normal(170, 10, 250)

y = []

for i in range(3):
    y.append(np.random.normal(170, 10, 250))

histogram(x, y, "test", [1, 2, 3])
