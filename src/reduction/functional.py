import numpy as np
from tqdm.auto import tqdm
from itertools import product

from ..reduction.clustering import optimize_with_clustering
from ..reduction.utils import get_modified_network_tep
from ..reduction.scenario_reduction import optimize_with_sr, ScenarioReduction
from ..optimization_models import tep, wind_investment

def reduce_optimize(
    opt_problem, problem_parameters, n_scenarios_range, model, model_params, n_vars, scenarios=None, cluster_center='centroid', scale=True, precompute_w=False
):
    assert scenarios is not None or opt_problem == "TEP", "Missing scenarios for non-TEP optimization problem"
    
    if opt_problem == "TEP":
        generation_scenarios = problem_parameters['RenewableProfiles']
        load_scenarios = problem_parameters['BusLoads']

        scenarios = np.empty(( 365, 24 * (len(generation_scenarios) + len(load_scenarios)) ), dtype=float)
        scenarios[:, :24 * len(generation_scenarios)] = generation_scenarios.transpose((1, 0, 2)).reshape(365, -1)
        scenarios[:, 24 * len(generation_scenarios):] = load_scenarios.transpose((1, 0, 2)).reshape(365, -1)
    
    results = np.empty((len(n_scenarios_range), n_vars))

    w = None
    if precompute_w:
        sr_tmp = ScenarioReduction(model_params, opt_problem, 1, problem_parameters, scale)
        if model_params == 'Morales':
            w = sr_tmp.morales_SR(scenarios) 
        elif model_params == 'Bruninx':
            w = sr_tmp.bruninx_SR(scenarios)

    for i, n_scenarios in enumerate(tqdm(n_scenarios_range)):
        results[i] = optimize_with_sr(scenarios, opt_problem, problem_parameters, n_scenarios, model_params, scale, w) if model.__name__ == "ScenarioReduction" else\
                     optimize_with_clustering(scenarios, opt_problem, problem_parameters, n_scenarios, model, model_params, cluster_center, scale)
    
    return results

def reduction_analysis(
    opt_problem, problem_parameters, n_scenarios_range, models, parameters_dict, n_vars, scenarios=None, cluster_representations=None, scale=True
):
    model_dict = {
        'centroid' : [],
        'medoid' : [],
        'sr' : []
    }

    if cluster_representations is None:
        print("Warning: since no cluster representation methods were given, clustering-based methods won't be applied.")

    if cluster_representations is not None:
        for cluster_representation in cluster_representations:
            for model in models:
                if model.__name__ != "ScenarioReduction":
                    model_parameter_combinations = [ dict(zip(parameters_dict[model.__name__].keys(), v)) for v in product(*parameters_dict[model.__name__].values()) ]
                    for model_parameters in model_parameter_combinations:
                        result = reduce_optimize(opt_problem, problem_parameters, n_scenarios_range, model, model_parameters, n_vars, scenarios, cluster_representation, scale)
                        
                        model_dict[cluster_representation].append(result)
    
    precompute_w = False
    for model in models:
        if model.__name__ == "ScenarioReduction":
            for method in parameters_dict[model.__name__]['method']:
                if method in ['Morales', 'Bruninx']:
                    precompute_w = True
                
                result = reduce_optimize(opt_problem, problem_parameters, n_scenarios_range, model, method, n_vars, scenarios, scale=scale, precompute_w=precompute_w)
                
                model_dict['sr'].append(result)
                precompute_c = False

    for model_type in ['centroid', 'medoid', 'sr']:
        if model_dict[model_type]:
            model_dict[model_type] = np.stack(model_dict[model_type], axis=0)

    return model_dict

def run_oos_test(
    problem_type, problem_parameters, reduction_results, num_clusters, x_range, fmax_range=None, name="oos"
):
    assert problem_type != "TEP" or fmax_range is not None, "Range for Fmax values need to be set for TEP OOS test"

    oos_results = {
        'centroid' : [],
        'medoid' : [],
        'sr' : []
    }
    
    for method_type in tqdm(reduction_results):
        n_methods = len(reduction_results[method_type])
        if n_methods > 0:
            oos_results[method_type] = np.zeros((n_methods, num_clusters))

            for n_m in tqdm(range(n_methods)):
                for n_cluster in tqdm(range(num_clusters)):
                    x = abs(reduction_results[method_type][n_m, n_cluster, :x_range])
                    if problem_type == "TEP":
                        fmax = reduction_results[method_type][n_m, n_cluster, x_range:fmax_range]
                        model_ss = tep.second_stage(x, fmax, **problem_parameters, name=name)
                        model_ss, _ = tep.solve_model(model_ss)
                    elif problem_type == "Wind Investment":
                        model_ss = wind_investment.second_stage(x, **problem_parameters)
                        model_ss = wind_investment.solve_model(model_ss)
                    
                    oos_results[method_type][n_m, n_cluster] = model_ss.getObjective().getValue()

    return oos_results