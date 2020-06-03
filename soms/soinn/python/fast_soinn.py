__author__ = 'robert'
import numpy as np
import numpy.matlib
from tqdm import tqdm
from tqdm import trange
import math

def distance(x, y):
    return np.linalg.norm((y - x))

# def learning(data=None, nepoch=None, spread_factor=0.1, age_max=None, lamb=None, c1=None, c2=None):
def learning(input_data=None, max_nepoch=None, spread_factor=0.1, lamb=None):
    """ A fast soinn function """
    # Initialize 2 nodes
    data = np.copy(input_data)
    nodes = data[0:2, :]
    m = np.array([1, 1])
    tem = distance(data[0, :], data[1, :])
    threshold = np.array([tem, tem])
    connection = np.array([[0, 0], [0, 0]])
    age = np.array([[0.0, 0.0], [0.0, 0.0]])
    nodes_point = np.array([0.0, 0.0])
    nodes_point0 = np.array([0.0, 0.0])
    nodes_point_valid_count = np.array([0, 0])
    nodes_density = np.array([0.0, 0.0])
    nodes_class_id = np.array([1, 0])

    value = np.array([0., 0.], dtype=np.float64)
    index = np.array([0, 0], dtype=np.int32)
    # sample_size = data.shape[0]
    sample_size = data.shape[0]*max_nepoch
    age_max = sample_size
    np.random.shuffle(data)
    total_newnodes = 0
    n_newnodes = 0
    for ii in trange(2, sample_size, desc='\tSOINN Progress', leave=True):
        # find winner node and runner-up node
        i = ii % data.shape[0]
        if i == 0:
            np.random.shuffle(data)
        dist = np.sqrt(np.sum(np.square(np.matlib.repmat(data[i, :], nodes.shape[0], 1) - nodes), axis=1))
        value[0] = np.amin(dist)
        index[0] = np.argmin(dist)
        dist[index[0]] = 1000000
        value[1] = np.amin(dist)
        index[1] = np.argmin(dist)
        # prototype, connection and age update
        if value[0] > threshold[index[0]] or value[1] > threshold[index[1]]:
            # add a new prototype
            nodes = np.concatenate((nodes, np.reshape(data[i, :], (1, -1))), axis=0)
            threshold = np.concatenate((threshold, np.array([1000000])))
            m = np.concatenate((m, np.array([1])))
            nodes_point = np.concatenate((nodes_point, np.array([0.0])))
            nodes_point0 = np.concatenate((nodes_point0, np.array([0.0])))
            nodes_point_valid_count = np.concatenate((nodes_point_valid_count, np.array([0])))
            nodes_density = np.concatenate((nodes_density, np.array([0.0])))
            nodes_class_id = np.concatenate((nodes_class_id, np.array([np.max(nodes_class_id) + 1])))
            connection = np.concatenate((connection, np.zeros((1, connection.shape[1]))), axis=0)
            connection = np.concatenate((connection, np.zeros((connection.shape[0], 1))), axis=1)
            age = np.concatenate((age, np.zeros((1, age.shape[1]))), axis=0)
            age = np.concatenate((age, np.zeros((age.shape[0], 1))), axis=1)
            n_newnodes += 1
        else:
            # find neighbor nodes of winner nodes
            neighbors = np.nonzero(connection[index[0], :])[0]
            if neighbors.shape[0] > 0:
                meandist = np.mean(dist[neighbors])
                nneighbors = np.sum(connection, axis=1)
                age[index[0], neighbors] += 1 + nneighbors[neighbors]*dist[neighbors]/meandist
                age[neighbors, index[0]] += 1 + nneighbors[neighbors]*dist[neighbors]/meandist
            # build connection
            if nodes_class_id[index[0]] == 0 or nodes_class_id[index[1]] == 0:
                connection[index[0], index[1]] = 1
                connection[index[1], index[0]] = 1
            elif nodes_class_id[index[0]] == nodes_class_id[index[1]]:
                connection[index[0], index[1]] = 1
                connection[index[1], index[0]] = 1
                nodes_class_id[index[1]] = nodes_class_id[index[0]]
            elif nodes_class_id[index[0]] != nodes_class_id[index[1]]:
                nodes_class_as_winner = (np.where(nodes_class_id == nodes_class_id[index[0]])[0])
                mean_density_class_winner = np.sum(nodes_density[nodes_class_as_winner]) / \
                                            np.float64(nodes_class_as_winner.shape[0])
                max_density_class_winner = np.max(nodes_density[nodes_class_as_winner])
                if 2.0 * mean_density_class_winner >= max_density_class_winner:
                    alpha_winner = 0.0
                elif 3.0 * mean_density_class_winner >= max_density_class_winner:
                    alpha_winner = 0.5
                else:
                    alpha_winner = 1.0
                # if min density of bmu1 and bmu2 is larger than max density in class --> can connect
                winner_condition = min(nodes_density[index[0]], nodes_density[index[1]]) >= \
                                   (alpha_winner * max_density_class_winner)

                nodes_class_as_runner = (np.where(nodes_class_id == nodes_class_id[index[1]])[0])
                mean_density_class_runner = np.sum(nodes_density[nodes_class_as_runner]) / \
                                            np.float64(nodes_class_as_runner.shape[0])
                max_density_class_runner = np.max(nodes_density[nodes_class_as_runner])
                if 2.0 * mean_density_class_runner >= max_density_class_runner:
                    alpha_runner = 0.0
                elif 3.0 * mean_density_class_runner >= max_density_class_runner:
                    alpha_runner = 0.5
                else:
                    alpha_runner = 1.0
                runner_condition = min(nodes_density[index[0]], nodes_density[index[1]]) > \
                                   (alpha_runner * max_density_class_runner)

                if winner_condition or runner_condition:  # winner or runner have high density to be treated as important node and maintain their connections
                    connection[index[0], index[1]] = 1
                    connection[index[1], index[0]] = 1
                    winner_runner_combine_class = min(nodes_class_id[index[0]], nodes_class_id[index[1]])
                    nodes_class_id[
                        nodes_class_as_winner] = winner_runner_combine_class  # winner and runner should be in the same class
                else:  # winner and runner have low density compared to others in class
                    connection[index[0], index[1]] = 0
                    connection[index[1], index[0]] = 0
                pass

            if connection[index[0], index[1]] == 1:
                combine_class = max(nodes_class_id[index[0]], nodes_class_id[index[1]])
                nodes_class_as_winner_runner = np.where(nodes_class_id == nodes_class_id[index[0]])[0]
                nodes_class_id[nodes_class_as_winner_runner] = combine_class
                nodes_class_as_winner_runner = np.where(nodes_class_id == nodes_class_id[index[1]])[0]
                nodes_class_id[nodes_class_as_winner_runner] = combine_class
            # age += 1.0
            age[index[1], index[0]] = 0.0
            age[index[0], index[1]] = 0.0
            neighbors = np.nonzero(connection[index[0], :])[0]
            # calculate the 'point'
            if neighbors.shape[0] > 0:
                winner_neighbor_diff = np.matlib.repmat(nodes[index[0], :], neighbors.shape[0], 1) - nodes[neighbors, :]
                winner_mean_distance = (1.0 / np.float64(neighbors.shape[0])) * \
                                       np.sum(np.sqrt(np.sum(np.square(winner_neighbor_diff), axis=1)))
                nodes_point[index[0]] += 1.0 / ((1.0 + winner_mean_distance) ** 2)
            # adjust the weight of winner node
            m[index[0]] += 1  # number of matches
            nodes[index[0], :] += (1.0 / np.float64(m[index[0]])) * (data[i, :] - nodes[index[0], :])
            if neighbors.shape[0] > 0:
                nodes[neighbors, :] += (1.0 / (100.0 * np.float64(m[index[0]]))) * \
                                       (np.matlib.repmat(data[i, :], neighbors.shape[0], 1) - nodes[neighbors, :])
            # delete the edges whose ages are greater than age_max
            if ii + input_data.shape[0] < sample_size:
                locate = np.where(age[index[0], :] > age_max)[0]
            else:
                locate = []
            connection[index[0], locate] = 0
            connection[locate, index[0]] = 0
            age[index[0], locate] = 0
            age[locate, index[0]] = 0

        # update threshold
        if np.count_nonzero(connection[index[0], :]) == 0:
            # no neighbor, the threshold should be the distance between winner node and runner-up node
            threshold[index[0]] = distance(nodes[index[0], :], nodes[index[1], :])
        else:
            # if have neighbors, choose the farthest one
            neighbors = np.nonzero(connection[index[0], :])[0]
            neighbor_distances = np.matlib.repmat(nodes[index[0], :], neighbors.shape[0], 1) - nodes[neighbors, :]
            threshold_winner = np.max(np.sqrt(np.sum(np.square(neighbor_distances), axis=1)))
            threshold[index[0]] = threshold_winner

        if np.count_nonzero(connection[index[1], :]) == 0:
            # no neighbor
            threshold[index[1]] = distance(nodes[index[0], :], nodes[index[1], :])
        else:
            neighbors = np.nonzero(connection[index[1], :])[0]
            neighbor_distances = np.matlib.repmat(nodes[index[1], :], neighbors.shape[0], 1) - nodes[neighbors, :]
            threshold_runner = np.max(np.sqrt(np.sum(np.square(neighbor_distances), axis=1)))
            threshold[index[1]] = threshold_runner
        # print(i, "/", sample_size)
        if (ii + 1) % lamb == 0 or (ii == sample_size - 1):  # or :
            # mean point
            nodes_point_sum_period = nodes_point - nodes_point0
            nodes_point_valid_node = np.nonzero(nodes_point_sum_period)[0]
            nodes_point_valid_count[nodes_point_valid_node] += 1
            for j in range(0, nodes_density.shape[0]):
                if nodes_point_valid_count[j] > 0:
                    nodes_density[j] = nodes_point[j] / nodes_point_valid_count[j]

            nodes_point0 = np.copy(nodes_point)

            # delete nodes with 0, 1 or 2 neighbors
            # mean_m = np.mean(m)
            # mean_density = np.mean(nodes_density)
            # min_density = np.min(nodes_density)
            # max_density = np.max(nodes_density)
            # std_density = np.std(nodes_density)
            q_density = np.quantile(nodes_density, 0.001/spread_factor)
            neighbors = np.sum(connection, axis=1)
            # neighbor0_set = np.intersect1d(np.where(m < mean_m)[0], np.where(neighbors == 0)[0])
            neighbor0_set = np.where(neighbors == 0)[0]
            # neighbor1_set = np.intersect1d(np.where(nodes_density < c1 * mean_density)[0], np.where(neighbors == 1)[0])
            neighbor1_set = np.where(neighbors == 1)[0]
            # c2 = 0.005/spread_factor
            # neighbor2_set = np.intersect1d(np.where(nodes_density < c2*mean_density)[0], np.where(neighbors == 2)[0])
            # neighbor2_set = np.intersect1d(np.where((mean_density-nodes_density) > math.pow(spread_factor, 2)*(mean_density-min_density))[0],
            #                                np.where(neighbors == 2)[0])
            # neighbor1_set = np.where(neighbors == 1)[0]
            neighbor2_set = np.intersect1d(np.where(nodes_density < q_density)[0],
                                           np.where(neighbors == 2)[0])
            if ii == sample_size - 1:
                # to_delete = np.array([])
                to_delete = neighbor0_set
                # to_delete = np.union1d(neighbor0_set, neighbor1_set)
            else:
                to_delete = np.union1d(neighbor0_set, neighbor1_set)
                to_delete = np.union1d(to_delete, neighbor2_set)
            if nodes.shape[0] - to_delete.shape[0] < 2:
                continue
            nodes = np.delete(nodes, to_delete, axis=0)
            threshold = np.delete(threshold, to_delete)
            m = np.delete(m, to_delete)
            nodes_point = np.delete(nodes_point, to_delete)
            nodes_point0 = np.delete(nodes_point0, to_delete)
            nodes_point_valid_count = np.delete(nodes_point_valid_count, to_delete)
            nodes_density = np.delete(nodes_density, to_delete)
            nodes_class_id = np.delete(nodes_class_id, to_delete)
            connection = np.delete(connection, to_delete, axis=0)
            connection = np.delete(connection, to_delete, axis=1)
            age = np.delete(age, to_delete, axis=0)
            age = np.delete(age, to_delete, axis=1)
            age_max = (lamb + nodes.shape[0])/2*spread_factor
            # age_max = lamb * spread_factor
            total_newnodes += n_newnodes
            n_newnodes = 0
        # if total_newnodes < data.shape[0]/10:
        #     break


    return nodes, connection, nodes_class_id
