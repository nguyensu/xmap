# Fusion based Genetic programming and SOM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu

c1 = np.random.rand(50, 3)/5
c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3)/5
c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3)/5
data = np.float32(np.concatenate((c1, c2, c3)))
colors = ["red"] * 50
colors.extend(["green"] * 50)
colors.extend(["blue"] * 50)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
labels = range(150)

n_rows, n_columns = 100, 160
som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid")
som.train(data)

som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)

c2_shifted = c2 - 0.2
updated_data = np.float32(np.concatenate((c1, c2_shifted, c3)))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(updated_data[:, 0], updated_data[:, 1], updated_data[:, 2], c=colors)

som.update_data(updated_data)
# som.train(epochs=10, radius0=20, scale0=0.02)
som.train()
som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)
#
