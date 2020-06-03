__author__ = 'robert'

import time
import numpy as np
import matplotlib.pyplot as plt
from soms.soinn.python import asc


mean1 = [0.5, 0.5]
mean2 = [3.0, 0.5]
conv1 = [[1, 0], [0, 1]]
conv2 = [[1.5, 0], [0, 1.2]]
class1_data = np.random.multivariate_normal(mean1, conv1, 15000)
class2_data = np.random.multivariate_normal(mean2, conv2, 15000)
train_data = np.concatenate((class1_data, class2_data))

plt.figure(1)
plt.plot(train_data[:, 0], train_data[:, 1], 'x')

start_time = time.time()
nodes, connection= asc.asoinn(train_data, 30, 100)
end_time = time.time()
print('soinn execute %s seconds' % (end_time - start_time))

plt.figure(2)
plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

for i in range(0, nodes.shape[0]):
    for j in range(0, nodes.shape[0]):
        if connection[i, j] != 0:
            plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-')
            pass

plt.show()
