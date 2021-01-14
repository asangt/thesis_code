import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import MaxAbsScaler

from ..reduction.utils import get_modified_network_tep, get_centroids, get_medoids
from ..optimization_models import tep, wind_investment

def cluster(scenarios, n_clusters, model, model_params, cluster_center='centroid', scale=True):
    get_center = get_centroids if cluster_center == 'centroid' else get_medoids

    if scale:
        scaler = MaxAbsScaler()
        scenarios = scaler.fit_transform(scenarios)

    model_instance = model(n_clusters, **model_params)
    labels = model_instance.fit_predict(scenarios)
    if scale:
        scenarios = scaler.inverse_transform(scenarios)
    
    clusters = model_instance.cluster_centers_ if cluster_center == 'custom' else get_center(scenarios, labels, n_clusters)
    if len(clusters.shape) > 2:
        clusters = clusters.squeeze(2)
    _, cluster_sizes = np.unique(labels, return_counts=True)
    rho = cluster_sizes / len(labels)
    
    return clusters, rho, labels

def optimize_with_clustering(
    scenarios, opt_problem, problem_parameters, n_clusters, model, model_params, cluster_center='centroid', scale=True
):
    supported_problems = ['TEP', 'Wind Investment']
    assert opt_problem in supported_problems, "Unsupported optimization problem"

    scenario_clusters, rho, labels = cluster(
        scenarios, n_clusters, model, model_params, cluster_center, scale
    )

    if opt_problem == "TEP":
        network = get_modified_network_tep(problem_parameters, scenarios, rho)

        model = tep.two_stage(**network)
        model, _ = tep.solve_model(model)

        x, f_max, cost = tep.get_from_model(model, len(network['Lines']), network['ExistingLinesNum'])
        result = np.array(x + f_max + [cost])
    elif opt_problem == "Wind Investment":
        model = wind_investment.two_stage(**problem_parameters, scenarios=scenario_clusters, scenario_w=rho)
        model = wind_investment.solve_model(model)

        x, cost = wind_investment.get_from_model(model)
        result = np.array([x] + [cost])
    
    return result