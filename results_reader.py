import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sb
to_load = sys.argv[1:]

results = []

for case in to_load:

    name = "TEST_" + case + "/results.npy"
    results.append(np.load(name))

x = np.arange(len(results))+1

mean = np.mean(results, axis=1)
min = np.min(results, axis=1)

for i, case in enumerate(to_load):

    print("\n------\n")

    print("Case:", case)
    print("\n")

    print("Mean:", round(mean[i], 5))
    print("Min:",  round(min[i], 5))


print("\n =================== \n")
print("\n =================== \n")

overall_min = round(np.min(min), 5)
case_min = to_load[np.argmin(min)]

print("Overall best result for case:", case_min)
print("Result:", overall_min)

argsort = np.argsort(min)

results_sorted = results[argsort]
cases_sorted = to_load[argsort]

for result, case in zip(results_sorted, cases_sorted):
    sb.histplot(result, label=case)

plt.show()