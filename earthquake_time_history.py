import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

earthquake = pd.read_csv("Excel/earthquake_time_history.csv").values

time_series = np.zeros([270, 2])
time_earthquake = 0

j = -1

for i in range(time_series.shape[0]):

    if time_earthquake < 2.9:
        # time += 0.1
        j += 10

    else:
        # time += 0.05
        j += 5

    time_earthquake = earthquake[j, 0]
    acceleration = earthquake[j, 1]

    time_series[i, 0] = time_earthquake
    time_series[i, 1] = acceleration

time_points = [47]



plt.scatter(time_series[:, 0], time_series[:, 1], s=10, c="dodgerblue")
plt.plot(time_series[:, 0], time_series[:, 1], c="dodgerblue")

for x in time_points:
    time = time_series[x, 0]
    label = "Time Marker: " + str(time) + " secs"
    plt.plot([time, time], [min(time_series[:, 1]) * 1.2, max(time_series[:, 1]) * 1.2],
             c='black', label=label)

plt.xlabel("Time From Earthquake Start (Secs)")
plt.ylabel(r"Earthquake Acceleration ($m/s^2$)")

plt.ylim([min(time_series[:, 1]) * 1.1, max(time_series[:, 1] * 1.1)])
plt.legend()

plt.show()
