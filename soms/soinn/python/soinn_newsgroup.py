__author__ = 'sunguyen'

from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
# from pprint import pprint
# pprint(list(newsgroups_train.target_names))

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from matplotlib import colors as mcolors
from soms.soinn.python import fast_soinn


categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
y = newsgroups_train.target


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = np.array(vectors.todense())
x_transformed = pca.fit_transform(X)

import matplotlib.pyplot as plt
cc = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
cc = [cc[c] for c in cc]
colt = [cc[i] for i in y]
plt.scatter(x_transformed[:,0], x_transformed[:,1], color=colt)
plt.show()

nodes, connection, classes = fast_soinn.learning(x_transformed, 50, 100, 1.5, 0.001)

def plot_soinn(nodes, connection):
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    for i in range(0, nodes.shape[0]):
        for j in range(0, nodes.shape[0]):
            if connection[i, j] != 0:
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-')
                pass
    plt.show()

# pca = PCA(n_components=2)
# # pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True, gamma=10)
# pca.fit(nodes)
# nodes = pca.transform(nodes)
plot_soinn(nodes, connection)
print(pca.explained_variance_ratio_)

print("Done!!")