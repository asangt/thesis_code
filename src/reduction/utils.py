import numpy as np
from scipy.spatial import distance

def correct_shapes(arr):
    if len(arr.shape) == 1:
        return arr[None, :]
    return arr

def kantorovich_distance(w_1, w_2, p):
    w_1 = correct_shapes(w_1)
    w_2 = correct_shapes(w_2)
        
    c = distance.cdist(w_1, w_2, metric='cityblock')
    d_k = (p @ c.min(axis=0)).sum()

    return d_k

def get_modified_network_tep(initial_network, scenarios, scenario_w=None):
    network = initial_network.copy()

    network['ScenariosNum'] = len(scenarios)
    network['BusLoads'] = scenarios[:, 24 * len(self.n_generation_scenarios):].reshape(self.n_scenarios, len(self.n_load_scenarios), 24).transpose((1, 0, 2))
    network['RenewableProfiles'] = scenarios[:, :24 * len(self.n_generation_scenarios)].reshape(self.n_scenarios, len(self.n_generation_scenarios), 24).transpose((1, 0, 2))
    if scenario_w is not None:
        network['ScenarioProbabilities'] = scenario_w.tolist()

    return network

def get_centroids(X, y, n_clusters):
    centroids = np.empty((n_clusters, X.shape[1]), dtype=float)
    centroid_w = np.empty(n_clusters, dtype=float)
    n_samples = len(X)
    
    for k in range(n_clusters):
        centroids[k] = X[y == k].mean(axis=0)
    
    return centroids

def get_medoids(X, y, n_clusters, metric='euclidean'):
    medoids = np.empty((n_clusters, X.shape[1]), dtype=float)
    medoid_w = np.empty(n_clusters, dtype=float)
    n_samples = len(X)
    
    for k in range(n_clusters):
        medoids[k] = X[y == k][distance.squareform(distance.pdist(X[y == k], metric)).sum(axis=1).argmin()]
    
    return medoids