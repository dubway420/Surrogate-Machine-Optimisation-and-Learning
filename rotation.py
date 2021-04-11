from machine_learning.dataset_generators import DatasetSingleFrame as Dataset
from machine_learning.dataset_generators import Cracks3D, Displacements, Cracks2D
import numpy as np

dataset = Dataset("/path/")

dataset = dataset

cracks3d = Cracks3D(dataset, array_type="position")


cracks3d_top = cracks3d.values[0, :, :, 0]

print(cracks3d_top.astype(int))

cracks2d = Cracks2D(dataset, array_type="position")

for row in cracks2d.values[0]:

    print(row)

# print(cracks1[:, :, 0].astype(int))
# print("\n --- \n")
# cracks1_new = np.rot90(cracks1, 1, (1, 0))
#
# print(cracks1_new[:, :, 0].astype(int))







