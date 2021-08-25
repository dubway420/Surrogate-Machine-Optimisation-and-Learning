from parmec_analysis.reactor_case import Parse as Case
import numpy as np

case = Case("batch21_7159_P40/batch21_7159_P40")

base_indices = case.results_indices

# for channel in base_indices:
#     print(channel)

indices_np = np.zeros([19, 19, 13])

first_numbers_row = case.first_numbers_row_interstitial
first_columns_row = case.first_columns_row_interstitial

number_columns = case.inter_columns

channel = 0

for row, column_offset in enumerate(first_columns_row):

    number_columns_this_row = number_columns - (2 * column_offset)
    for column in range(column_offset, column_offset + number_columns_this_row):

        channel_value = np.array(base_indices[channel][0:13])
        indices_np[row, column] = channel_value
        channel += 1


for row in indices_np[:, :, 12].tolist():
    print(row)

print("====")

indices_rotated = np.rot90(indices_np, 1, (1, 0))

for row in indices_rotated[:, :, 12].tolist():
    print(row)





