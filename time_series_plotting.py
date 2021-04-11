from parmec_analysis.reactor_case import ParmecInstance
import matplotlib.pyplot as plt
import numpy as np

time_series = np.zeros(271)
time = 0

for i in range(time_series.shape[0]):

    if time < 2.9:
        time += 0.1
    else:
        time += 0.05

    time_series[i] = time

test_case = ParmecInstance("batch21_7159_P40/batch21_7159_P40")

print(test_case.get_id())
fig = plt.figure(figsize=(10, 10))
axs = [plt.subplot(2, 1, 1), plt.subplot(2, 1, 2)]

axs[0].set_xlabel("Time From Earthquake Start (Secs)")
axs[0].set_ylabel("Maximum channel x displacement (mm)")

result = np.array(test_case.result_time_history())*1000

xy_pos = test_case.get_brick_xyz_positions(include='xy')

channels = [34, 55, 61, 76, 116, 130, 154, 183, 273, 305]

axs[1].scatter(xy_pos[0], xy_pos[1], color='grey', alpha=0.5, s=100)
#
for channel in channels:

    label = "Channel" + str(channel)
    axs[0].plot(time_series, result[:, channel-1], label=label)

    axs[1].scatter(xy_pos[0][channel-1], xy_pos[1][channel-1], s=100)


axs[1].set(adjustable='box-forced', aspect='equal')

axs[1].set_xticklabels([])
axs[1].set_yticklabels([])

fig.tight_layout()

plt.show()

print("Time series in seconds:")
print(time_series)


