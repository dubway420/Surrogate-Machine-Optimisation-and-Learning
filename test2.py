from parmec_analysis.reactor_case import Parse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

instance_intact = Parse('intact_core_rb')

intact_coords = instance_intact.get_brick_xyz_positions('xy', channel_type='inter')

unaugmented1 = Parse("C:/Users/Huw/Documents/DataAug/Unaugmented/unaugmented")

unaugmented1.apply_augmentation("flip_3")

unaugmented_crack_top = unaugmented1.get_crack_array(levels=4, print_mode=True)

unaugmented_displacements = unaugmented1.get_result_at_time(result_type='all', time_index=48, result_columns="2")

# print(unaugmented_displacements[:, -1] * 1000)
#
# print("\n====================\n")

rotated_m90 = Parse("C:/Users/Huw/Documents/DataAug/flipVert2/flipVert2")

rotated_m90_crack_top = rotated_m90.get_crack_array(levels=4, print_mode=True)

rotated_m90_displacements = rotated_m90.get_result_at_time(result_type='all', time_index=48, result_columns="2")

# print(rotated_m90_displacements[:, -1] * 1000)
#
# print(unaugmented_crack_top == rotated_m90_crack_top)

#######################################################
##
#######################################################

fig = plt.figure(figsize=(12, 8))

# Generates the plot grid for each case
counts_grid = AxesGrid(fig, 111,
                       # There's a row for each epoch data plus one for the ground truth labels
                       # There's a column for each case to be plotted
                       nrows_ncols=(1, 2),
                       axes_pad=0.3,
                       cbar_mode='single',
                       cbar_location='bottom',
                       cbar_pad=0.2
                       )

im = counts_grid[0].scatter(intact_coords[0], intact_coords[1],
            marker='o', c=unaugmented_displacements[:, -1], cmap='jet', label='inter',
            s=30)

im.set_clim(np.amin(unaugmented_displacements[:, -1]), np.amax(unaugmented_displacements[:, -1]))

im = counts_grid[1].scatter(intact_coords[0], intact_coords[1],
            marker='o', c=rotated_m90_displacements[:, -1], cmap='jet', label='inter',
            s=30)

im.set_clim(np.amin(rotated_m90_displacements[:, -1]), np.amax(rotated_m90_displacements[:, -1]))

counts_grid.cbar_axes[0].colorbar(im)

plt.show()

# counts_grid[1].scatter(intact_coords[0], intact_coords[1],
#             marker='o', c=rotated_m90_displacements[:, -1], cmap='jet', label='inter',
#             s=30)

# plt.show()
