import numpy as np

array = np.array(
    [
        [34, 56, 5848, 454],
        [45, 54, 65, 433],
        [563, 87, 54, 345]
    ])


print(np.mean(array, axis=1))
