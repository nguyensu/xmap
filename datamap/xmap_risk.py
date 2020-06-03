# data mapping and analytics platform (DataMAP)
import pandas as pd
import time
import os.path
import umap
import matplotlib.pyplot as plt
import warnings
import numpy as np
from soms.soinn.python import fast_soinn
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.spatial import ConvexHull
from sklearn.preprocessing import PolynomialFeatures
from enum import Enum
from sklearn.preprocessing import MinMaxScaler
import pickle
warnings.simplefilter('ignore')

NNK, NS, SEED, f, cmap, NFEATURE = None, None, None, None, None, None

class STEP(Enum):
    INITIALIZING = 0
    DATA_CLEANED = 1
    UMAP_CLEANED = 2
    SOINN_CLEANED = 3
    CONTEXT_EXPLAINED = 4

def plot_soinn(nodes, connection):
    plt.tight_layout()
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')
    for i in range(0, nodes.shape[0]):
        for j in range(0, nodes.shape[0]):
            if connection[i, j] != 0:
                plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-')
                pass
    plt.axis('off')
    plt.show()

def print_output(*argv, ee="\n"):
    print(argv[0], end=ee)
    print(argv[0], end=ee, file=f)
    f.flush()

def plot_embedding_space(embedding, labels=None, label_index=None, lnames=None, data_name=""):
    plt.rc('legend', **{'fontsize': 15})
    plt.figure(figsize=(14, 14))
    # plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    sizes = []
    for li in range(len(label_index)):
        msize = 5
        label = label_index[li]
        points = embedding[labels == label]
        if points.shape[0] != 0:
            cname = lnames[li] + " Size=" + str(points.shape[0])
            sizes.append(points.shape[0])
            if len(label_index) > 2 and points.shape[0] > 2 and "highlight" in data_name:
                hull = ConvexHull(points)
                area = hull.area
                density = np.round(points.shape[0]/area, 2)
                if "highlightX" not in data_name:
                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], color=cmap[li])
                else:
                    cname = cname.replace("# 0", "NOMATCH").replace("# 100", "CONFUSED")
                    msize = 50
                if "NOMATCH" not in cname:
                    plt.text(np.median(points[::, 0]),
                             np.median(points[::, 1]),
                             lnames[li].split()[1], fontsize=30, fontweight="bold",
                             bbox={'facecolor': cmap[li], 'alpha': 0.5, 'edgecolor': 'none', 'boxstyle': 'round'},
                             color="white", alpha=0.7)
                # plt.fill(points[hull.simplices, 0], points[hull.simplices, 1], color=cmap[li], alpha=0.3)
                # plt.text(0.5*(np.max(points[hull.simplices, 0]) + np.min(points[hull.simplices, 0])),
                #          0.5*(np.max(points[hull.simplices, 1]) + np.min(points[hull.simplices, 1])),
                cname += " Area=" + str(np.round(area, 2)) + " Denst=" + str(density)
            plt.scatter(
                points[::, 0], points[::, 1], c=cmap[li],
                s=msize, label=cname, alpha=0.5
            )
    plt.legend()#(bbox_to_anchor=(0.4,0.8), loc="upper right")  # plt.colorbar()
    plt.gca().legend(markerscale=5 if msize == 5 else 1.0)
    plt.title(data_name + " SIZE#{} DIM#{} ({:.2f}/{:.2f})".format(labels.shape[0], NFEATURE,
                                                                 (labels.shape[0] - labels.sum()) / labels.shape[0],
                                                                 labels.sum() / labels.shape[0]) + "\numap_k{}_ns{}_s{}".format(
        NNK, NS, SEED), fontsize=20)
    plt.axis('off')
    plt.savefig("outputs/umap_k{}_ns{}_s{}_".format(NNK, NS, SEED) + data_name + ".png")
    plt.show()
    return sizes

def run_xmap(dataset=None, n_neighbors=None, negative_sample_rate=None, seed=None, model_folder="models"):
    global f, SEED, NNK, NS, cmap, NFEATURE
    SEED = seed
    NNK = n_neighbors
    NS = negative_sample_rate
    np.random.seed(SEED)
    pathname = model_folder + "/" + "xmap_k{}_ns{}_s{}_".format(NNK, NS, SEED) + dataset
    if os.path.exists(pathname + ".notes"):
        step = pickle.load(open(pathname + ".notes", "rb"))
    else:
        step = STEP.INITIALIZING

    if step == STEP.INITIALIZING:
        f = open("outputs/xmap_k{}_ns{}_s{}_".format(NNK, NS, SEED) + dataset + ".log", 'w')
    else:
        f = open("outputs/xmap_k{}_ns{}_s{}_".format(NNK, NS, SEED) + dataset + ".log", 'w+')

    t0 = time.time()
    print_output("Loading Data ...")
    print_output("\tData set: {}".format(dataset))
    if step.value < STEP.DATA_CLEANED.value:
        data = pd.read_csv("data/{}.csv".format(dataset))
        feature_names = [c.replace("_", "ubar").replace(".", "dot") for c in data.columns][1:]
        nfeatures = len(feature_names)
        NFEATURE = nfeatures
        data = data.values
        Y = data[:, 0].reshape(-1, 1)
        X = data[:, 1:]

        scaler = MinMaxScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X)
        # umap
        step = STEP.DATA_CLEANED
        pickle.dump(step, open(pathname + ".notes", "wb"))
        pickle.dump((X_norm, Y, scaler, nfeatures, feature_names), open(pathname + ".cleandata", "wb"))
    else:
        print_output("\tLoad cleaned data from " + pathname + ".cleandata")
        X_norm, Y, scaler, nfeatures, feature_names = pickle.load(open(pathname + ".cleandata", "rb"))

    print_output("Learning UMAP ...")
    if step.value < step.UMAP_CLEANED.value:
        reducer = umap.UMAP(random_state=SEED, n_neighbors=NNK, negative_sample_rate=NS)
        reducer.fit(X_norm)
        step = STEP.UMAP_CLEANED
        pickle.dump(step, open(pathname + ".notes", "wb"))
        pickle.dump(reducer, open(pathname + ".umap", "wb"))
    else:
        print_output("\tLoad trained umap from " + pathname + ".umap")
        reducer = pickle.load(open(pathname + ".umap", "rb"))
    embedding = reducer.transform(X_norm)

    # fig, ax = plt.subplots(figsize=(6, 5))
    lnames = ["Negative", "Positive"]
    Y = Y.reshape(-1)
    # color = [Y[i] for i in range(X.shape[0])]
    cmap = ["blue", "red",  "purple", "hotpink", "black", "green", "orange", "teal", "brown",
            "lightsteelblue", "gray", "lime", "coral", "plum", "gold", "c", "tomato", "blueviolet",
            "darkseagreen"]
    plot_embedding_space(embedding, labels=Y, label_index=[0, 1], lnames=lnames, data_name="gt_"+dataset)

    # nepoch: number of times the data passed to SOINN; age_max: maximum age of a connection; age increases if the
    # connection (different from the second best) links to the BMU. If max_age is too small the topological relationships will
    # be prematurely destroyed. Meanwhile if max_age is too large, some useless connections may survive because of randomness
    # or noise --> SOINN needs to run longer to get the accurate results and more relationships will be preserved.
    # lamb: is the number of steps (or number of processed inputs) before SOINN checks and cleans up the network. Lambda has
    # a similar effect as compared to max_age, i.e. small lamb leads to unstable network (unable to establish topological
    # relationhips) while large lamb may lead to redundant nodes and connections.
    print_output("Learning topological relations and Determining contexts ....")
    if step.value < step.SOINN_CLEANED.value:
        lamb = 500
        # if data.shape[0] < lamb:
        #     lamb = data.shape[0]
        nodes, connection, classes = fast_soinn.learning(input_data=embedding, max_nepoch=5, spread_factor=1.0, lamb=lamb)

        classes = 0*classes
        cmap = cmap*10
        plot_soinn(nodes, connection)

        G = nx.Graph()
        for i in range(0, nodes.shape[0]):
            for j in range(0, nodes.shape[0]):
                if connection[i, j] != 0:
                    G.add_edge(i, j, weight=1.0)

        network_cd_alg = "best"
        n_components = nx.number_connected_components(G)
        max_context = int(n_components + np.sqrt(n_components))
        threshold = 0.2
        if network_cd_alg == "gn":
            from networkx.algorithms import community
            communities_generator = community.girvan_newman(G)
            # for _ in range(max(max_context - n_components, 0)):
            #     level_communities = next(communities_generator)
            while True:
                level_communities = next(communities_generator)
                size_com = [len(c) for c in level_communities if len(c) > 1]
                if min(size_com) < threshold*sum([len(c) for c in level_communities]):
                    break
            cms = sorted(map(sorted, level_communities))
        elif network_cd_alg == "best" or network_cd_alg == "dendo":
            import community
            if network_cd_alg == "best":
                cms = community.best_partition(G)
            else:
                dendrogram = community.generate_dendrogram(G)
                sized = len(dendrogram)
                cms = community.partition_at_level(dendrogram, sized-2)
            coms = set([cms[i] for i in cms])
            cdict = {}
            for k in coms:
                cdict[k] = []
            for i in cms:
                cdict[cms[i]].append(i)
            cms = []
            for k in coms:
                cms.append(list(cdict[k]))
        else:
            # cms = list(nx.connected_components(G))
            from networkx.algorithms import community
            communities_generator = community.girvan_newman(G)
            level_communities = next(communities_generator)
            cms = sorted(map(sorted, level_communities))

        components = [c for c in cms if len(c) > 1]
        # print(components)
        count = 1
        for comp in components:
            for n in comp:
                classes[n] = count
            count += 1
        nclusters = len(components)
        step = STEP.SOINN_CLEANED
        pickle.dump(step, open(pathname + ".notes", "wb"))
        pickle.dump((nodes, connection, classes, nclusters), open(pathname + ".soinn", "wb"))
    else:
        print_output("\tLoad trained umap from " + pathname + ".soinn")
        nodes, connection, classes, nclusters = pickle.load(open(pathname + ".soinn", "rb"))
    # classes = [cid[c] for c in classes]
    cid = [c for c in range(nclusters+1)]
    cid = cid + [100]
    if step.value < step.CONTEXT_EXPLAINED.value:
        nbrs = NearestNeighbors(n_neighbors=1).fit(nodes)
        distances, indices = nbrs.kneighbors(embedding)
        indices = list(indices.reshape(-1))
        indices = np.array([classes[indices[i]] for i in range(len(indices))])
        cmap = 10*["red", "blue", "purple", "hotpink", "black", "green", "orange", "teal", "brown",
                "lightsteelblue", "gray", "lime", "coral", "plum", "gold", "c", "tomato", "blueviolet",
                "darkseagreen"]
        plot_embedding_space(embedding, labels=indices, label_index=cid, lnames=["context__# "+str(c) for c in cid], data_name="pointcontext_" + dataset)
        cluster_sizes = plot_embedding_space(embedding, labels=indices, label_index=cid, lnames=["context__# " + str(c) for c in cid],
                             data_name="highlightcontext_" + dataset)

        cluster_id_ranked_by_size = (-np.array(cluster_sizes)).argsort()
        poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        cluster_explainer_dict = {}
        if nclusters > 1:
            # intepret the context
            print_output("Explaining contexts ...")
            xcluster_id = np.zeros(embedding.shape[0])
            xcluster_id_details = np.zeros((embedding.shape[0], nclusters))
            outputs = np.zeros((nclusters, len(feature_names)))
            cluster_characteristic_dict = {}
            feature_names_I = None
            finteraction = False
            XX = X_norm
            feature_names = [ff.replace("ubar", "_").replace("dot", ".") for ff in feature_names]
            if finteraction:
                XX = poly.fit_transform(1 - X_norm)
                XX = 1 - XX
                feature_names_I = str(poly.get_feature_names())
                for fi in range(nfeatures):
                    feature_names_I = feature_names_I.replace("'x" + str(fi) + "'", feature_names[fi])
                    feature_names_I = feature_names_I.replace("'x" + str(fi) + " ", feature_names[fi] + " ")
                    feature_names_I = feature_names_I.replace(" x" + str(fi) + "'", " " + feature_names[fi])
                feature_names_I = feature_names_I.replace("[", "").replace("]", "").replace("'", "").replace(", ",
                        ",").replace(" ", " or ")
                feature_names_I = feature_names_I.split(",")
                feature_names = feature_names_I
                outputs = np.zeros((nclusters, len(feature_names)))
            for i in range(nclusters):
                cluster_id = i #cluster_id_ranked_by_size[i]
                n_identity_feature = 10
                print_output("Context #" + str(cluster_id+1))
                Xc = XX[indices == cluster_id+1]
                for fi in range(len(feature_names)):
                    outputs[cluster_id][fi] = np.std(Xc[::, fi])
                true_features = []
                false_features = []
                numeric_features = []
                impure_features = []
                ranked_features = np.argsort(outputs[cluster_id])
                for fi in ranked_features:
                    if outputs[cluster_id][fi] == 0:
                        val = np.unique(Xc[::, fi])
                        # (values, counts) = np.unique(Xc[::,fi], return_counts=True)
                        # ind = np.argmax(counts)
                        # val = values[ind]
                        if val == 1.0:
                            true_features.append(fi)
                        elif val == 0.0:
                            false_features.append(fi)
                        else:
                            numeric_features.append(fi)
                    else:
                        impure_features.append((fi, np.min(Xc[::, fi]), np.max(Xc[::, fi]), np.average(Xc[::, fi])))
                cluster_explainer_dict[cluster_id] = (finteraction, true_features, false_features, numeric_features, impure_features)
                cluster_characteristic_dict[cluster_id] = [true_features, false_features, numeric_features]
                nzeros = len(feature_names) - np.count_nonzero(outputs[cluster_id])
                mask = np.ones((X.shape[0], ), dtype=bool)
                if nzeros > n_identity_feature:
                    n_identity_feature = nzeros
                countf = 0
                print_output("\tTrue Features")
                count = 0
                for fi in true_features:
                    if countf > n_identity_feature:
                        break
                    countf += 1
                    count += 1
                    fmask = XX[::, fi] == 1.0
                    mask = mask & fmask
                if count > 0:
                    print_output("\t\t" + str(sorted([feature_names[ii] for ii in true_features[:count]])))
                print_output("\tFalse Features")
                count = 0
                for fi in false_features:
                    if countf > n_identity_feature:
                        break
                    countf += 1
                    count += 1
                    fmask = XX[::, fi] == 0.0
                    mask = mask & fmask
                if count > 0:
                    print_output("\t\t" + str(sorted([feature_names[ii] for ii in false_features[:count]])))
                print_output("\tNumeric Features")
                count = 0
                for fi in numeric_features:
                    if countf > n_identity_feature:
                        break
                    countf += 1
                    count += 1
                if count > 0:
                    print_output("\t\t" + str([(feature_names[ii[0]], ii[1], ii[2]) for ii in numeric_features[:count]]))
                # xcluster_id = mask*(cluster_id+1)
                # xcluster_id = np.where(xcluster_id == 0,
                #                        mask*(cluster_id+1),
                #                        np.where(mask != 0, 100, xcluster_id))
                xcluster_id_details[mask, cluster_id] = 1
                print_output("\t" + 20 * '-')
                # print_output("# Zeros Std. = {}".format(nzeros))
            print_output("\t" + 20 * '=')
            print_output("")

        step = STEP.CONTEXT_EXPLAINED
        pickle.dump(step, open(pathname + ".notes", "wb"))
        pickle.dump((cluster_explainer_dict, xcluster_id_details), open(pathname + ".xcluster", "wb"))
    else:
        print_output("\tLoad explainer from " + pathname + ".xcluster")
        cluster_explainer_dict, xcluster_id_details = pickle.load(open(pathname + ".xcluster", "rb"))

    for i in range(nclusters):
        plot_embedding_space(embedding, labels=xcluster_id_details[::, i], label_index=cid,
                             lnames=["xcontext__# " + str(c) for c in cid],
                             data_name="highlightXcontext_" + str(i + 1) + "   __" + dataset)

    run_time = time.time() - t0
    print_output('Run in %.3f s' % run_time)

    print_output("Complete!!!")

if __name__ == "__main__":
    datasets = ["german_data", "bank_data", "spambase_data", "mushroom_data", "breastcancer_data", "adult_data",
                "australian_data", "mammo_data"]
    run_xmap(dataset="german_data", n_neighbors=15, negative_sample_rate=5, seed=2)