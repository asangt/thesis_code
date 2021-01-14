import numpy as np
import gurobipy as gp
from gurobipy import GRB

def two_stage(c, c_inv, c_ws, c_ls, Pmax, scenarios, scenario_w, s_base=100):
    m = gp.Model("Wind investment")
    
    demand = scenarios[:, 0].copy() * s_base
    wind_generation = scenarios[:, 1].copy()
    
    # Variables
    x = m.addVar(vtype=GRB.CONTINUOUS, lb=0., name="x")
    w_sp = m.addVars(len(scenarios), vtype=GRB.CONTINUOUS, lb=0., name="w_sp")
    l_sh = m.addVars(len(scenarios), vtype=GRB.CONTINUOUS, lb=0., name="l_sh")
    g = m.addVars(len(scenarios), vtype=GRB.CONTINUOUS, lb=0., name="Pc")
    
    # Constraints
    EnergyBalanceConstraint = m.addConstrs(
        (x * wind_generation[s] + g[s] - w_sp[s] + l_sh[s] == demand[s]
         for s in range(len(scenarios))), name='energy_balance'
    )
    
    ConventionalGeneratorUB = m.addConstrs(
        (g[s] <= Pmax for s in range(len(scenarios))),\
        name="cg_ub"
    )
    
    WindSpillagetUB = m.addConstrs(
        (w_sp[s] <= x * wind_generation[s] for s in range(len(scenarios))),\
        name="ws_ub"
    )
    
    LoadSheddingUB = m.addConstrs(
        (l_sh[s] <= demand[s] for s in range(len(scenarios))),\
        name="ls_ub"
    )
    
    OperationCosts = gp.quicksum(
        scenario_w[s] * ( c * g[s] + w_sp[s] * c_ws + l_sh[s] * c_ls )\
        for s in range(len(scenarios))
    )
    
    m.setObjective(
        x * c_inv + OperationCosts, sense=GRB.MINIMIZE
    )
    
    return m

def second_stage(x, c, c_inv, c_ws, c_ls, Pmax, scenarios, scenario_w, s_base=100):
    m = gp.Model("Wind Investment - SS only")
    
    demand = scenarios[:, 0].copy() * s_base
    wind_generation = scenarios[:, 1].copy()
    
    # Variables
    w_sp = m.addVars(len(scenarios), vtype=GRB.CONTINUOUS, lb=0., name="w_sp")
    l_sh = m.addVars(len(scenarios), vtype=GRB.CONTINUOUS, lb=0., name="l_sh")
    g = m.addVars(len(scenarios), vtype=GRB.CONTINUOUS, lb=0., name="Pc")
    
    # Constraints
    EnergyBalanceConstraint = m.addConstrs(
        (x * wind_generation[s] + g[s] - w_sp[s] + l_sh[s] == demand[s]
         for s in range(len(scenarios))), name='energy_balance'
    )
    
    ConventionalGeneratorUB = m.addConstrs(
        (g[s] <= Pmax for s in range(len(scenarios))),\
        name="cg_ub"
    )
    
    WindSpillagetUB = m.addConstrs(
        (w_sp[s] <= x * wind_generation[s] for s in range(len(scenarios))),\
        name="ws_ub"
    )
    
    LoadSheddingUB = m.addConstrs(
        (l_sh[s] <= demand[s] for s in range(len(scenarios))),\
        name="ls_ub"
    )
    
    OperationCosts = gp.quicksum(
        scenario_w[s] * ( c * g[s] + w_sp[s] * c_ws + l_sh[s] * c_ls )\
        for s in range(len(scenarios))
    )
    
    m.setObjective(
        x * c_inv + OperationCosts, sense=GRB.MINIMIZE
    )
    
    return m

def solve_model(model, verbose=False):
    model.Params.OutputFlag = 1 if verbose else 0
    model.optimize()
    
    return model

def get_from_model(model):
    model_x = model.x[0]
    model_cost = model.getObjective().getValue()

    return model_x, model_cost