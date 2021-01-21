import numpy as np

def instantiate_network(buses, scenarios, lines, line_mileage_cost, ref_node, big_constant=None, susceptancy_scaling=1.0, scenario_weights=None):
    """
        buses: {'bus'  : [has_generation, num_conv, num_ren, c_op, conv_cap, ren_cap, ren_type, ls_cost, load]}
        lines: {'line' : [length, x, capacity, is_built, count, is_candidate, corridor_capacity]}
    """
    
    BusesNum     = len(buses)
    ScenariosNum = scenarios.shape[0]
    
    LineExisting = [ k for (k, v) in lines.items() if v[3] ]
    LineCandidates = [ k for (k, v) in lines.items() if not v[3] and v[5] ]
    Lines = LineExisting + LineCandidates
    ExistingLinesNum = len(LineExisting)
    
    LineSusceptancies  = [ (1.0 / lines[l][1]) * susceptancy_scaling for l in Lines ]
    MinLineCapacities  = [ lines[le][2] * lines[le][4] for le in LineExisting ]
    MinLineCapacities += [ 0.2 * lines[lc][2] for lc in LineCandidates ]
    MaxLineCapacities  = [ lines[l][-1] * lines[l][2] for l in Lines ]
    
    ExpansionCosts = [ lines[l][0] * line_mileage_cost / lines[l][2] for l in Lines ]
    BaseInvestmentCosts = [ 0.4 * lines[lc][0] * line_mileage_cost for lc in LineCandidates ]
    
    ConventionalUnits = []
    RenewableUnits    = []
    LoadDistribution  = []
    for (k, v) in buses.items():
        LoadDistribution += [v[-1]]
        if v[0]:
            if v[1] > 0:
                ConventionalUnits += [k] * v[1]
            if v[2] > 0:
                RenewableUnits += [k] * v[2]
    
    BusLoads = np.empty((BusesNum, ScenariosNum, 24), dtype=float)
    for i, d in enumerate(LoadDistribution):
        BusLoads[i] = scenarios[:, 48:] * d
    
    OperationalCosts  = [ buses[g][3] for g in ConventionalUnits ]
    LoadSheddingCosts = [ v[-2] for (k,v) in buses.items() ]
    ConventionalGeneratorLimits = [ buses[g][4] for g in ConventionalUnits ]
    RenewableGenerationLimits = [ buses[g][5] for g in RenewableUnits ]
    RenewableProfiles = np.empty((len(RenewableUnits), ScenariosNum, 24), dtype=float)
    for i, r in enumerate(RenewableUnits):
        start = 0 if buses[r][-3] == 'wind' else 24
        RenewableProfiles[i] = scenarios[:, start:start+24] * RenewableGenerationLimits[i]
    ReferenceNode = ref_node
    
    ScenarioProbabilities = scenario_weights if scenario_weights is not None else [1.0] * ScenariosNum
    if type(ScenarioProbabilities) == list:
        ScenarioProbabilities = np.array(ScenarioProbabilities)
    
    M = BusLoads.max() * 2 if big_constant is None else big_constant
    
    Network = {
        "BusesNum" : BusesNum,
        "ScenariosNum" : ScenariosNum,
        "Lines" : Lines,
        "ExistingLinesNum" : ExistingLinesNum,
        "ConventionalUnits" : ConventionalUnits,
        "RenewableUnits" : RenewableUnits,
        "BusLoads" : BusLoads,
        "BaseInvestmentCosts" : BaseInvestmentCosts,
        "ExpansionCosts" : ExpansionCosts,
        "OperationalCosts" : OperationalCosts,
        "LoadSheddingCosts" : LoadSheddingCosts,
        "LineSusceptancies" : LineSusceptancies,
        "MinLineCapacities" : MinLineCapacities,
        "MaxLineCapacities" : MaxLineCapacities,
        "ConventionalGeneratorLimits" : ConventionalGeneratorLimits,
        "RenewableProfiles" : RenewableProfiles,
        "ReferenceNode" : ReferenceNode,
        "ScenarioProbabilities" : ScenarioProbabilities,
        "M" : M
    }
    
    return Network