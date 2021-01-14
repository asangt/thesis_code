import numpy as np
import gurobipy as gp
from gurobipy import GRB

def two_stage(BusesNum, ScenariosNum, Lines, ExistingLinesNum, ConventionalUnits, RenewableUnits, BusLoads,\
              BaseInvestmentCosts, ExpansionCosts, OperationalCosts, LoadSheddingCosts, LineSusceptancies,\
              MaxLineCapacities, MinLineCapacities, ConventionalGeneratorLimits, RenewableProfiles, ReferenceNode, ScenarioProbabilities, M, name="TEP", save=False):
    """
    Inputs:
      BusesNum - total number of buses in the network,
      ScenariosNum - total number of available uncertainty scenarios,
      Lines - set of tranmission lines,
      ExistingLinesNum - number of existing transmission lines,
      ConventionalUnits - set of conventional generator units,
      RenewableUnits - set of renewable generator units
      BusLoads - load profiles for the network,
      BaseInvestmentCosts - investment costs for candidate lines that are not dependent on capacity,
      ExpansionCosts - investment costs for candidate lines that depend on the capacity,
      OperationalCosts - operational costs for running conventional generation units,
      LoadSheddingCosts - costs for load shedding,
      LineSusceptancies - line susceptancies,
      MaxLineCapacities - maximum possible line capacities,
      MinLineCapacities - minimum possible line capacities,
      ConventionalGeneratorLimits - minimum and maximum generation limits for the conventional generators,
      RenewableProfiles - profiles for renewable generators,
      ReferenceNode - specifying the reference bus,
      ScenarioProbabilities - probabilities for each scenario to happen,
      M - large constant
    """
    
    m = gp.Model(name)
    
    # Indices
    ScenarioIndices = range(ScenariosNum)
    TimeIndices     = range(24)
    
    LeIndices   = range(ExistingLinesNum)
    LcIndices   = range(ExistingLinesNum, len(Lines))
    LineIndices = range(len(Lines))
    Buses       = range(1, BusesNum + 1)
    CgIndices   = range(len(ConventionalUnits))
    RgIndices   = range(len(RenewableUnits))
    
    # Variables
    x     = m.addVars(LineIndices, vtype=GRB.BINARY, name="x")
    f_max = m.addVars(LineIndices, vtype=GRB.CONTINUOUS, lb=0, name="f_max")
    Ls    = m.addVars(Buses, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=0, name="Ls")
    Pc    = m.addVars(CgIndices, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=0, name="Pc")
    Pr    = m.addVars(RgIndices, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=0, name="Pr")
    f     = m.addVars(LineIndices, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=float('-inf'), name="f")
    theta = m.addVars(Buses, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=float('-inf'), name="theta")
    
    # Constraints
    ExistingLinesConstraint = m.addConstrs( (x[i] == 1 for i in LeIndices), name='x_le' )
    ReferenceAngleConstraint = m.addConstrs(
        (theta[ReferenceNode, s, t] == 0 for s in ScenarioIndices for t in TimeIndices), name='theta_ref'
    )
    
    EnergyBalanceConstraint = m.addConstrs(
        (gp.quicksum(Pc[g, s, t] for g in CgIndices if n == ConventionalUnits[g]) +\
        gp.quicksum(Pr[r, s, t] for r in RgIndices if n == RenewableUnits[r]) +\
        gp.quicksum(f[l, s, t] for l in LineIndices if n == Lines[l][1]) -\
        gp.quicksum(f[l, s, t] for l in LineIndices if n == Lines[l][0]) == BusLoads[n-1, s, t] - Ls[n, s, t]\
        for n in Buses for s in ScenarioIndices for t in TimeIndices), name='energy_balance'
    )
    
    PowerFlowConstraintLB = m.addConstrs(
        (-(1 - x[l]) * M <= f[l, s, t] - LineSusceptancies[l] * (theta[Lines[l][0], s, t] - theta[Lines[l][1], s, t])\
        for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_lb'
    )
    
    PowerFlowConstraintUB = m.addConstrs(
        (f[l, s, t] - LineSusceptancies[l] * (theta[Lines[l][0], s, t] - theta[Lines[l][1], s, t]) <= (1 - x[l]) * M\
        for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_ub'
    )
    
    LineCapacityConstraintLB = m.addConstrs(
        (x[l] * MinLineCapacities[l] <= f_max[l] for l in LineIndices),\
        name="lc_limits_lb"
    )
    
    LineCapacityConstraintUB = m.addConstrs(
        (f_max[l] <= x[l] * MaxLineCapacities[l] for l in LineIndices),\
        name="lc_limits_ub"
    )
    
    PowerFlowConstraintLB_2 = m.addConstrs(
        (-f_max[l] <= f[l, s, t] for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_lb'
    )
    
    PowerFlowConstraintUB_2 = m.addConstrs(
        (f[l, s, t] <= f_max[l] for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_ub'
    )
    
    ConventionalGeneratorLimitUB = m.addConstrs(
        (Pc[g, s, t] <= ConventionalGeneratorLimits[g] for g in CgIndices for s in ScenarioIndices for t in TimeIndices),\
        name="cg_limits_ub"
    )
    
    RenewableGeneratorLimitUB = m.addConstrs(
        (Pr[r, s, t] <= RenewableProfiles[r, s, t] for r in RgIndices for s in ScenarioIndices for t in TimeIndices),\
        name="rg_limits_ub"
    )
    
    LoadSheddingLimitUB = m.addConstrs(
        (Ls[n, s, t] <= BusLoads[n-1, s, t] for n in Buses for s in ScenarioIndices for t in TimeIndices),\
        name="ls_limits_ub"
    )
    
    NewLineInvestmentCosts = gp.quicksum(
        BaseInvestmentCosts[lc - ExistingLinesNum] * x[lc] for lc in LcIndices
    )
    
    LineExpansionCosts = gp.quicksum(
        ExpansionCosts[l] * (f_max[l] - MinLineCapacities[l]) for l in LineIndices
    )
    
    AnnualOperationalCosts = gp.quicksum( 
        365 * ScenarioProbabilities[s] * gp.quicksum( gp.quicksum( OperationalCosts[g] * Pc[g, s, t] for g in CgIndices ) +\
                                                gp.quicksum( LoadSheddingCosts[n-1] * Ls[n, s, t] for n in Buses )\
                                                for t in TimeIndices )\
        for s in ScenarioIndices
    )
    
    m.setObjective(
        NewLineInvestmentCosts + LineExpansionCosts + AnnualOperationalCosts, sense=GRB.MINIMIZE
    )
    
    if save:
        m.write('{}.mps'.format(name))
    
    return m

def second_stage(x, f_max, BusesNum, ScenariosNum, Lines, ExistingLinesNum, ConventionalUnits, RenewableUnits, BusLoads, \
                 BaseInvestmentCosts, ExpansionCosts, OperationalCosts, LoadSheddingCosts, LineSusceptancies, \
                 MaxLineCapacities, MinLineCapacities, ConventionalGeneratorLimits, RenewableProfiles, ReferenceNode, ScenarioProbabilities, M, name="SS-TEP"):

    m = gp.Model(name)
    x = [1.0] * ExistingLinesNum + x.tolist()
    f_max = f_max.tolist()
    
    # Indices
    ScenarioIndices = range(ScenariosNum)
    TimeIndices     = range(24)
    
    LeIndices   = range(ExistingLinesNum)
    LcIndices   = range(ExistingLinesNum, len(Lines))
    LineIndices = range(len(Lines))
    Buses       = range(1, BusesNum + 1)
    CgIndices   = range(len(ConventionalUnits))
    RgIndices   = range(len(RenewableUnits))
    
    # Variables
    Ls    = m.addVars(Buses, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=0, name="Ls")
    Pc    = m.addVars(CgIndices, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=0, name="Pc")
    Pr    = m.addVars(RgIndices, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=0, name="Pr")
    f     = m.addVars(LineIndices, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=float('-inf'), name="f")
    theta = m.addVars(Buses, ScenarioIndices, TimeIndices, vtype=GRB.CONTINUOUS, lb=float('-inf'), name="theta")
    
    # Constraints
    ReferenceAngleConstraint = m.addConstrs(
        (theta[ReferenceNode, s, t] == 0 for s in ScenarioIndices for t in TimeIndices), name='theta_ref'
    )
    
    EnergyBalanceConstraint = m.addConstrs(
        (gp.quicksum(Pc[g, s, t] for g in CgIndices if n == ConventionalUnits[g]) +\
        gp.quicksum(Pr[r, s, t] for r in RgIndices if n == RenewableUnits[r]) +\
        gp.quicksum(f[l, s, t] for l in LineIndices if n == Lines[l][1]) -\
        gp.quicksum(f[l, s, t] for l in LineIndices if n == Lines[l][0]) == BusLoads[n-1, s, t] - Ls[n, s, t]\
        for n in Buses for s in ScenarioIndices for t in TimeIndices), name='energy_balance'
    )
    
    PowerFlowConstraintLB = m.addConstrs(
        (-(1 - x[l]) * M <= f[l, s, t] - LineSusceptancies[l] * (theta[Lines[l][0], s, t] - theta[Lines[l][1], s, t])\
        for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_lb'
    )
    
    PowerFlowConstraintUB = m.addConstrs(
        (f[l, s, t] - LineSusceptancies[l] * (theta[Lines[l][0], s, t] - theta[Lines[l][1], s, t]) <= (1 - x[l]) * M\
        for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_ub'
    )
    
    PowerFlowConstraintLB_2 = m.addConstrs(
        (-f_max[l] <= f[l, s, t] for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_lb'
    )
    
    PowerFlowConstraintUB_2 = m.addConstrs(
        (f[l, s, t] <= f_max[l] for l in LineIndices for s in ScenarioIndices for t in TimeIndices), name='pf_limits_ub'
    )
    
    ConventionalGeneratorLimitUB = m.addConstrs(
        (Pc[g, s, t] <= ConventionalGeneratorLimits[g] for g in CgIndices for s in ScenarioIndices for t in TimeIndices),\
        name="cg_limits_ub"
    )
    
    RenewableGeneratorLimitUB = m.addConstrs(
        (Pr[r, s, t] <= RenewableProfiles[r, s, t] for r in RgIndices for s in ScenarioIndices for t in TimeIndices),\
        name="rg_limits_ub"
    )
    
    LoadSheddingLimitUB = m.addConstrs(
        (Ls[n, s, t] <= BusLoads[n-1, s, t] for n in Buses for s in ScenarioIndices for t in TimeIndices),\
        name="ls_limits_ub"
    )
    
    NewLineInvestmentCosts = gp.quicksum(
        BaseInvestmentCosts[lc - ExistingLinesNum] * x[lc] for lc in LcIndices
    )
    
    LineExpansionCosts = gp.quicksum(
        ExpansionCosts[l] * (f_max[l] - MinLineCapacities[l]) for l in LineIndices
    )
    
    AnnualOperationalCosts = gp.quicksum( 
        365 * ScenarioProbabilities[s] * gp.quicksum( gp.quicksum( OperationalCosts[g] * Pc[g, s, t] for g in CgIndices ) +\
                                                gp.quicksum( LoadSheddingCosts[n-1] * Ls[n, s, t] for n in Buses )\
                                                for t in TimeIndices )\
        for s in ScenarioIndices
    )
    
    m.setObjective(
        NewLineInvestmentCosts + LineExpansionCosts + AnnualOperationalCosts, sense=GRB.MINIMIZE
    )
    
    return m

def solve_model(model, solve_fixed=False, verbose=False):
    model.Params.OutputFlag = 1 if verbose else 0
    model.optimize()
    
    fixed_model = None
    if solve_fixed:
        fixed_model = model.fixed()
        fixed_model.optimize()
    
    return model, fixed_model

def get_from_model(m, n_lines, n_existing):
    model_x = np.array(m.x[n_existing:n_lines])
    model_fmax = np.array(m.x[n_lines:n_lines + n_lines])
    model_cost = m.getObjective().getValue()
    
    return model_x, model_fmax, model_cost