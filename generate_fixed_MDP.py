import sys, getopt
import pickle
import uuid
from finite_deterministic_mdp import FiniteDMDP

"""
Takes as input 5 parameters (S, A, H, t, path) and creates a MDP with the specified size,
and whose minimum value gap is larger than t. Then it stores the MDP in a .pkl file in folder path.
"""
argumentList = sys.argv[1:]
 
# Options
options = "n:S:A:H:t:p:"
 
# Long options
long_options = ["nMDPs", "Nstates", "Nactions", "Threshold", "Horizon", "Path"]
 
try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
    H = 0
    S = 0
    A = 0
    th = 0
    path = ""
    # checking each argument
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-n", "--nMDPs"):
            n = int(currentValue)
        elif currentArgument in ("-S", "--Nstates"):
            S = int(currentValue)
        elif currentArgument in ("-A", "--Nactions"):
            A = int(currentValue)
        elif currentArgument in ("-H", "--Horizon"):
            H = int(currentValue)
        elif currentArgument in ("-t", "--Threshold"):
            th = float(currentValue)
        elif currentArgument in ("-p", "--Path"):
            path = str(currentValue)
             
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

for i in range(n): #FiniteDMDP.RandomwithGapThreshold(
    env = FiniteDMDP.Random(                                          
        H,
        S,
        A,
        Rmax = 1,
        initial_state_distr=0,
        sigma = 0.5)
    minGap = 0.1
    minGap = round(minGap,4)
    idx = uuid.uuid4() #random identifier of the MDP
    with open(path+f"S{S}A{A}H{H}_{idx}_Delta{minGap}.pkl","wb") as file:
        pickle.dump(env,file)