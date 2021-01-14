import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MaxAbsScaler
import gurobipy as gp
from gurobipy import GRB

from ..optimization_models import tep, wind_investment
from ..reduction.utils import kantorovich_distance, get_modified_network_tep

class ScenarioReduction:
    
    def __init__(self, method, opt_problem, n_scenarios=5, problem_parameters=None, scale=True):
        self.supported_methods = ['Dupacova', 'Morales', 'Bruninx']
        self.supported_problems = ['TEP', 'Wind Investment']

        assert method in self.supported_methods, "Unknown method"
        assert opt_problem in self.supported_problems, "Unknown optimization problem"
        assert problem_parameters is not None or method == 'Dupacova', "Problem (network) parameters are required for methods other than Dupacova"

        self.method = method
        self.opt_problem = opt_problem
        self.n_scenarios = n_scenarios
        self.problem_parameters = problem_parameters
        self.scale = scale
        if scale:
            self.scaler = MaxAbsScaler()

        if self.opt_problem == "TEP":
            self.n_generation_scenarios = len(network['RenewableProfiles'])
            self.n_load_scenarios = len(network['BusLoads'])

    def solve_DP(self, X):
        EX = X.mean(axis=0)[None, :]

        if self.opt_problem == "TEP":
            network = get_modified_network_tep(self.problem_parameters, EX, np.ones(1))

            model = tep.two_stage(**network)
            model, _ = tep.solve_model(model)
            
            x_dp, fmax_dp = tep.get_from_model(model)
            return (x_dp, fmax_dp)
        elif self.opt_problem == "Wind Investment":
            model = wind_investment.two_stage(**self.problem_parameters, scenarios=EX, scenario_w=np.ones(1))
            model = wind_investment.solve_model(model)

            x_dp = model.x[0]
            return (x_dp, )
    
    def solve_SS(self, scenarios, scenario_w, var_dp):
        if self.opt_problem == "TEP":
            network = get_modified_network_tep(self.problem_parameters, scenarios, scenario_w)
            
            x_dp, fmax_dp = var_dp[0], var_dp[1]
            model_ss = tep.second_stage(**network, x=x_dp, f_max=fmax_dp)
            model_ss, _ = tep.solve_model(model_ss)
        elif self.opt_problem == "Wind Investment":
            x_dp = var_dp[0]

            model_ss = wind_investment.second_stage(**self.problem_parameters, scenarios=scenarios, scenario_w=scenario_w, x=x_dp)
            model_ss = wind_investment.solve_model(model_ss)
        
        cost = model_ss.getObjective().getValue()
        return cost
    
    def solve_TS(self, scenarios, scenario_w):
        if self.opt_problem == "TEP":
            network = get_modified_network_tep(self.problem_parameters, scenarios, scenario_w)

            model = tep.two_stage(**network)
            model, _ = tep.solve_model(model)
        elif self.opt_problem == "Wind Investment":
            model = wind_investment.two_stage(**self.problem_parameters, scenarios=scenarios, scenario_w=scenario_w)
            model = wind_investment.solve_model(model)
        
        cost = model.getObjective().getValue()
        return cost
    
    def morales_SR(self, X):
        # solve deterministic expected value problem
        var_dp = self.solve_DP(X)
        
        # solve second stage with single scenarios
        Z = np.empty((len(X), 1), dtype=float)
        for i in range(len(Z)):
            w = X[i, :][None, :].copy()
            scenario_w = np.ones(len(w))
            Z[i] = self.solve_SS(w, scenario_w, var_dp)
        
        return Z

    def bruninx_SR(self, X):
        Z = np.empty((len(X), 1), dtype=float)
        for i in range(len(Z)):
            w = X[i, :][None, :].copy()
            scenario_w = np.ones(len(w))
            Z[i] = self.solve_TS(w, scenario_w)
        
        return Z
    
    def get_optimal_weights_sr(self, X, u, J, p):
        q = np.empty(len(u), dtype=float)
        labels = np.empty(len(X), dtype=int)
        labels[u] = np.arange(len(u))
        
        j_mins = distance.cdist(X[u], X[J], metric='cityblock').argmin(axis=0)
        for j in range(len(u)):
            j_i = J[(j_mins == j).nonzero()[0]]
            q[j] = p[j] + p[j_i].sum()
            labels[j_i] = j
        
        return q, labels

    def forward_selection(self, X):
        w = X.copy()
        N, n_features = X.shape
        w_r = np.empty((self.n_scenarios, n_features), dtype=float)
        
        p = np.ones(len(w)) / len(w)
        
        for j in range(self.n_scenarios):
            u_dist = np.empty(len(w), dtype=float)
            for i in range(len(w)):
                w_1 = np.insert(w_r[:j], j, w[i], axis=0)
                w_2 = np.delete(w, i, axis=0)
                u_dist[i] = kantorovich_distance(w_1, w_2, p[:N-j-1])
            
            idx = u_dist.argmin()
            w_r[j] = w[idx]
            w = np.delete(w, idx, axis=0)
        
        u = np.empty(self.n_scenarios, dtype=int)
        for j in range(self.n_scenarios):
            u[j] = np.where(np.all(w_r[j] == X, axis=1))[0]

        J = np.delete(np.arange(len(X)), u, axis=0)
        q, labels = self.get_optimal_weights_sr(X, u, J, p)

        return u, q, labels

    def fit(self, X):

        if self.method == 'Morales':
            w = self.morales_SR(X)
        elif self.method == 'Bruninx':
            w = self.bruninx_SR(X)
        else:
            w = self.scaler.fit_transform(X) if self.scale else X

        u, q, labels = self.forward_selection(w)
        w_r = X[u]

        self.labels_ = labels
        self.reduced_set_ = w_r
        self.reduced_weights_ = q

        return self

    def fit_return(self, X):
        self.fit(X)

        return self.reduced_set_, self.reduced_weights_, self.labels_

def optimize_with_sr(
    scenarios, opt_problem, problem_parameters, n_scenarios, method, scale=True
):
    model_instance = ScenarioReduction(method, opt_problem, n_scenarios, problem_parameters, scale)
    reduced_set, rho, labels = model_instance.fit_return(scenarios)

    if opt_problem == "TEP":
        network = get_modified_network_tep(problem_parameters, reduced_set, rho)

        opt_model = tep.two_stage(**network)
        opt_model, _ = solve_model(opt_model)

        x, f_max, cost = tep.get_from_model(opt_model, len(network['Lines']), network['ExistingLinesNum'])
        result = np.array(x + f_max + [cost])
    elif opt_problem == "Wind Investment":
        opt_model = wind_investment.two_stage(**problem_parameters, scenarios=reduced_set, scenario_w=rho)
        opt_model = wind_investment.solve_model(opt_model)

        x, cost = wind_investment.get_from_model(opt_model)
        result = np.array([x] + [cost])
    
    return result