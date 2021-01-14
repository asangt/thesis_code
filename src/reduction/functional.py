import numpy as np
from tqdm.auto import tqdm
from itertools import product

from ..reduction.clustering import optimize_with_clustering
from ..reduction.utils import get_modified_network_tep
from ..reduction.scenario_reduction import optimize_with_sr

def reduce_optimize(
    opt_problem, problem_parameters, n_scenarios_range, model, model_params, n_vars, scenarios=None, cluster_center='centroid', scale=True
):
    assert scenarios is not None or opt_problem == "TEP", "Missing scenarios for non-TEP optimization problem"
    
    if opt_problem == "TEP":
        generation_scenarios = problem_parameters['RenewableProfiles']
        load_scenarios = problem_parameters['BusLoads']

        scenarios = np.empty(( 365, 24 * (len(generation_scenarios) + len(load_scenarios)) ), dtype=float)
        scenarios[:, :24 * len(generation_scenarios)] = generation_scenarios.transpose((1, 0, 2)).reshape(365, -1)
        scenarios[:, 24 * len(generation_scenarios):] = load_scenarios.transpose((1, 0, 2)).reshape(365, -1)
    
    results = np.empty((len(n_scenarios_range), n_vars))

    for i, n_scenarios in enumerate(tqdm(n_scenarios_range)):
        results[i] = optimize_with_sr(scenarios, opt_problem, problem_parameters, n_scenarios, model_params, scale) if model.__name__ == "ScenarioReduction" else\
                     optimize_with_clustering(scenarios, opt_problem, problem_parameters, n_scenarios, model, model_params, cluster_center, scale)
    
    return results

def reduction_analysis(
    opt_problem, problem_parameters, n_scenarios_range, models, parameters_dict, n_vars, scenarios=None, cluster_representations=None, scale=True
):
    centroid_models, medoid_models, sr_models = [], [], []

    if cluster_representations is None:
        print("Warning: since no cluster representation methods were given, clustering based methods won't be applied.")

    if cluster_representations is not None:
        for cluster_representation in cluster_representations:
            for model in models:
                if model.__name__ != "ScenarioReduction":
                    model_parameter_combinations = [ dict(zip(parameters_dict[model.__name__].keys(), v)) for v in product(*parameters_dict[model.__name__].values()) ]
                    for model_parameters in model_parameter_combinations:
                        result = reduce_optimize(opt_problem, problem_parameters, n_scenarios_range, model, model_parameters, n_vars, scenarios, cluster_representation, scale)
                        
                        if cluster_representation == 'centroid':
                            centroid_models.append(result)
                        elif cluster_representation == 'medoid':
                            medoid_models.append(result)
    
    for model in models:
        if model.__name__ == "ScenarioReduction":
            for method in parameters_dict[model.__name__]['method']:
                result = reduce_optimize(opt_problem, problem_parameters, n_scenarios_range, model, method, n_vars, scenarios, scale=scale)

                sr_models.append(result)
    
    centroid_models, medoid_models, sr_models = np.stack(centroid_models, axis=0), np.stack(medoid_models, axis=0), np.stack(sr_models, axis=0)
    reduced_models = np.concatenate((centroid_models, medoid_models, sr_models), axis=0)

    return reduced_models