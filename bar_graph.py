import numpy as np
import matplotlib.pyplot as plt

# The number of grouped columns
cases = ('Case 341', 'Case 123')
N = len(cases)

# The number of bars in each group
Epochs = ("Ground Truth", "Epoch 32", "Epoch 54")

# This generates some dummy data data for the graph. Replace this with your data
data = np.random.randint(5, 20, (len(Epochs), len(cases)))

# Generates dimensional metrics for the graph
ind = np.arange(N)
width = 1/(N+len(Epochs)/2)

# Iterates through the data and plots the bars
for i, epoch in enumerate(data):
    pos = ind + (i * width)
    plt.bar(pos, epoch, width, label=Epochs[i])

plt.ylabel('Loss')
plt.xticks(ind + width, cases)
plt.legend(loc='best')
plt.show()


#
# plt.title('Scores by group and gender')
#
#
#
# plt.show()