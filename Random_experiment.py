from EPRL import EPRLAgent
from BPI_UCRL import BPI_UCRLAgent
from plot_utils import process_results, plot
from rlberry.manager import AgentManager
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def get_MDP(path):
    file = open(path, 'rb')
    env = pickle.load(file)
    file.close()
    return env

if __name__ == '__main__':
    #Experiment parameters
    n_MDPs = 1000 #number of random MDPs
    n_sim = 1 # 1 simulation per MDP
    max_budget = 1e7
    #Environment parameters
    size = "S2A2H3" #size of the MDPs
    S = int(size[1])
    A = int(size[3])
    H = int(size[5])
    folder = "./instances/random/"
    #generate the rando MDPs
    os.system(f"python3 generate_fixed_MDP.py -n {n_MDPs} -S {S} -A {A} -H {H} -t 0.1 -p {folder} ")
    
    MDP_names = [] #list of MDP identifiers
    Gaps = [] #list of corresponding minimum Gaps
    count = 0
    for file_name in os.listdir("./instances/random/"):
        if file_name[:6] == size and count < n_MDPs:
            concat = file_name.split(".pkl")[0].split("_Delta")
            idx = concat[0].split(size+"_")[1]
            minGap = float(concat[1])
            MDP_names.append(idx)
            Gaps.append(minGap)
            count+= 1
    env_ctor = get_MDP #environmment constructor : fetches a MDP that was already constructed by generate_fixed_MDPs
    
    S = int(size.split("S")[1][0])
    A = int(size.split("A")[1][0])
    H = int(size.split("H")[1][0])
    kwargs = dict(
    S = S,
    A = A,
    H = H
    )

    #Algorithm meta-parameters (shared by all algorithms)
    epsilon = 0.1 #redefined later
    delta = 0.1 #confidence parameter

    #Agents parameters
    maxD_params = {
        "sampling_rule": "max_diameter",
        "delta": delta,
        "epsilon": epsilon,
        "stage_dependent": True,
        "period" : None,
    }

    AmaxCov_params = {
        "sampling_rule": "adaptive_max_coverage",
        "delta": delta,
        "epsilon": epsilon,
        "stage_dependent": True,
        "period" : None,
    }

    BPI_UCRL_params = {
        "delta": delta,
        "epsilon": epsilon,
        "stage_dependent": True,
        "period" : None,
    }

    
    for j, idx in enumerate(MDP_names):
        # DataFrame to record data
        minGap = Gaps[j] # minimum value gap of the MDP
        epsilon = 0.05*minGap
        path = "./instances/random/"+size+"_"+idx+"_Delta"+str(minGap)+".pkl"
        env_kwargs = dict(path = path)
        
        maxD_params["epsilon"] = epsilon
        AmaxCov_params["epsilon"] = epsilon
        BPI_UCRL_params["epsilon"] = epsilon
        # Create AgentManager to fit n_sim agents using multiple jobs
        maxD = AgentManager(
            EPRLAgent,
            train_env = (env_ctor, env_kwargs),
            fit_budget=max_budget,
            init_kwargs=maxD_params,
            n_fit=n_sim,
            agent_name ="maxD",
            parallelization="process",
            output_dir="./results",
        )


        AmaxCov = AgentManager(
            EPRLAgent,
            (env_ctor, env_kwargs),
            fit_budget=max_budget,
            init_kwargs=AmaxCov_params,
            n_fit=n_sim,
            agent_name ="AmaxCov",
            parallelization="process",
            output_dir="./results",
        )

        BPI_UCRL = AgentManager(
            BPI_UCRLAgent,
            train_env = (env_ctor, env_kwargs),
            fit_budget= max_budget,
            init_kwargs=BPI_UCRL_params,
            n_fit=n_sim,
            agent_name ="BPI_UCRL",
            parallelization="process",
            output_dir="./results",
        )
        agents = [maxD, AmaxCov, BPI_UCRL]
        for agent in agents:
            agent.fit()
        names = {str(maxD) : "maxD", str(AmaxCov) : "AmaxCov", str(BPI_UCRL) : "BPI_UCRL"}
        #plot results for each MDP
        #plot(agents, names, kwargs, variable ="max_diameter")
        #plot(agents, names, kwargs, variable = "avg_actions")
        #Taus = plot(agents, names, kwargs, variable ="tau")
        #Correct = plot(agents, names, kwargs, variable ="correct")
        df = process_results(agents, tag="max_diameter")
        df1 = process_results(agents, tag= "Qstar_diameter")
        correct = process_results(agents, tag="correct") # check correctness of the algos
        df["Qstar_diameter"] = df1["Qstar_diameter"] 
        df["MDP"] = idx
        correct["MDP"] = idx
        with open(f"./results/logs/random/"+size+"_"+idx+".pkl","wb") as file:
            pickle.dump([df, correct, minGap], file)
            