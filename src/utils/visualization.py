import numpy as np
import matplotlib.pyplot as plt

def cluster_plots(method_type, methods, reduction_results, true_cost, num_clusters_range, cost_location=-1, figsize=(8, 6), save_name=None):
    plt.figure(figsize=figsize)
    for n_m, method in enumerate(methods):
        plt.plot(num_clusters_range, reduction_results[method_type][n_m, :, cost_location] / true_cost, label=method)
    title = method_type.capitalize() if method_type != 'sr' else "Scenario reduction"
    plt.title("{} methods".format(title), fontsize=14)
    plt.ylabel("Relative cost", fontsize=14)
    plt.xlabel("Number of clusters", fontsize=14)
    plt.axhline(1.0, color='r', ls='--', label='full data')
    plt.legend()
    
    if save_name:
        plt.savefig(save_name, dpi=400)
    
    plt.show()