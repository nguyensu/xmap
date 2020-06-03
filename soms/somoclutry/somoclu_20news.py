# Fusion based Genetic programming and SOM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
# from pprint import pprint
# pprint(list(newsgroups_train.target_names))

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from matplotlib import colors as mcolors
from soms.soinn.python import fast_soinn
cc = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
cc = [cc[c] for c in cc]


categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
y = newsgroups_train.target
colt = [cc[i] for i in y]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = np.array(vectors.todense())
x_transformed = pca.fit_transform(X)

labels = y

n_rows, n_columns = 100, 160
som = somoclu.Somoclu(n_columns, n_rows, maptype="toroid")
som.train(X)

som.view_umatrix(bestmatches=True, bestmatchcolors=colt, labels=labels)
