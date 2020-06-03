__author__ = 'robert'

import time
# import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from soms.soinn.python import fast_soinn
import numpy as np

def plot_soinn(nodes, connection):
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    for i in range(0, nodes.shape[0]):
        for j in range(0, nodes.shape[0]):
            if connection[i, j] != 0:
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-')
                pass
    plt.show()

# plt.switch_backend('Qt4Agg')

start_time = time.time()
train_data_file = 'soinn_demo_train.mat'
train_data = sio.loadmat(train_data_file)
train = train_data['train']
end_time = time.time()
print('loading train data executes %s seconds' % (end_time - start_time))

plt.figure(1)
plt.plot(train[:, 0], train[:, 1], 'bo')
plt.show()

start_time = time.time()
# nepoch: number of times the data passed to SOINN; age_max: maximum age of a connection; age increases if the
# connection (different from the second best) links to the BMU. If max_age is too small the topological relationships will
# be prematurely destroyed. Meanwhile if max_age is too large, some useless connections may survive because of randomness
# or noise --> SOINN needs to run longer to get the accurate results and more relationships will be preserved.
# lamb: is the number of steps (or number of processed inputs) before SOINN checks and cleans up the network. Lambda has
# a similar effect as compared to max_age, i.e. small lamb leads to unstable network (unable to establish topological
# relationhips) while large lamb may lead to redundant nodes and connections.

nodes, connection, classes = fast_soinn.learning(input_data=train, max_nepoch=2, spread_factor=1, lamb=500)

# print(indices)
# import networkx as nx
# G=nx.Graph()
#
# for i in range(0, nodes.shape[0]):
#     for j in range(0, nodes.shape[0]):
#         if connection[i, j] != 0:
#             G.add_edge(i, j, weight=1.0)
# nx.draw(G)
# plt.show()

# print("Number of components = ", nx.number_connected_components(G))

end_time = time.time()
print('classes is %s' % classes)
print('soinn execute %s seconds' % (end_time - start_time))

plot_soinn(nodes, connection)
#
# from sklearn.neighbors import NearestNeighbors
# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(nodes)
# distances, indices = nbrs.kneighbors(train)


#
# # traformation before visualisation
# from sklearn.decomposition import PCA, KernelPCA
# pca = PCA(n_components=2)
# # pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True, gamma=10)
# pca.fit(nodes)
# nodes = pca.transform(nodes)
# plot_soinn(nodes, connection)
# print(pca.explained_variance_ratio_)