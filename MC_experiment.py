from EPRL import EPRLAgent
from BPI_UCRL import BPI_UCRLAgent
from multiprocessing import Process, Manager
import os
import pickle
import pandas as pd


def get_MDP(path):
    file = open(path, 'rb')
    env = pickle.load(file)
    file.close()
    return env


def OneSimu(Id, sim, agent_name, agent_params, max_budget, path, MDPid, data):
    """
    Id : int 
        index of the process that will execute the function.
    sim : int
        index of the simulation number.
    agent_name : str
    agent_params : dictionary
        contains the parameters that are needed by the __init__() method of the agent's class.
    max_budget : int
        maximum number of episodes to run the agent's fit() method.
    path : str
        path to the .pkl file containing the environnment.
    MDPid : str 
        Unique indentifier of the environnement (used in case of multiple synthetic MDPs).
    data : list
        Used to store dataframes summarizing results of each Monte-Carlo experiment.
    """
    env = get_MDP(path)
    if agent_name in ["maxD", "AmaxCov"]:
        agent = EPRLAgent(env, **agent_params)
        agent.fit(max_budget)
        
    elif agent_name == "BPI_UCRL":
        agent = BPI_UCRLAgent(env, **agent_params)
        agent.fit(max_budget)
        
    df = pd.DataFrame(agent.writer.__dict__["_data"]["max_diameter"])
    df = df.rename(columns = {"value": "max_diameter", "global_step" : "episode"})
    df = df.drop(columns = ["tag"])
    df1 = pd.DataFrame(agent.writer.__dict__["_data"]["Qstar_diameter"])
    df1 = df1.rename(columns = {"value": "Qstar_diameter", "global_step" : "episode"}) 

    correct = pd.DataFrame(agent.writer.__dict__["_data"]["correct"])
    correct = correct.rename(columns = {"value" : "correct"})
    correct = correct.drop(columns = ["tag"])
    df["n_simu"] = sim
    df["Qstar_diameter"] = df1["Qstar_diameter"]
    df["MDP"] = MDPid
    df["name"] = agent_name
    
    correct["MDP"] = MDPid
    correct["n_simu"] = sim
    correct["name"] = agent_name
    data[Id] = (df, correct)
    return 

    

if __name__ == '__main__':
    #Experiment parameters
    n_sim = 10
    max_budget = 1e7
    done = []
    for file_name in os.listdir("./results/logs/"):
        concat = file_name.split(".pkl")[0]
        idx = concat.split("_")[1]
        done.append(idx)
        
    #Environment parameters
    size = "S2A2H3" #size of the MDPs
    MDP_names = [] #list of MDP identifiers
    Gaps = [] #list of corresponding minimum Gaps
    for file_name in os.listdir("./instances/"):
        if file_name.split("_")[0] == size:
            concat = file_name.split(".pkl")[0].split("_Delta")
            idx = concat[0].split(size+"_")[1]
            minGap = float(concat[1])
            if not (idx in done):
                MDP_names.append(idx)
                Gaps.append(minGap)
    
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

    parameters = {"maxD": maxD_params, "AmaxCov": AmaxCov_params, "BPI_UCRL": BPI_UCRL_params}
    for j, MDPid in enumerate(MDP_names):
        minGap = Gaps[j] # minimum value gap of the MDP
        epsilon = 0.05*minGap
        path = "./instances/"+size+"_"+MDPid+"_Delta"+str(minGap)+".pkl"
        maxD_params["epsilon"] = epsilon
        AmaxCov_params["epsilon"] = epsilon
        BPI_UCRL_params["epsilon"] = epsilon
        
        #Running Parallel Monte-Carlo simulations
        jobs= []
        data = Manager().list(range(n_sim*len(parameters.keys()))) #list to record data of each simulation
        pId = 0 # id of the process
        for agent_name in parameters.keys():
            for sim in range(n_sim):
                p = Process(target=OneSimu,
                            args = (pId, sim,
                                    agent_name, parameters[agent_name], max_budget,
                                    path, MDPid,
                                    data))
                p.start()
                print(f"{agent_name} n° {sim} started")
                jobs.append(p)
                pId += 1
        for p in jobs:
            p.join()
        #concatenating and storing the results of all algorithms in a .pkl file
        df = pd.DataFrame()
        correct = pd.DataFrame()
        for i in range(len(data)):
            dfi , correcti = data[i]
            df = pd.concat([df, dfi])
            correct = pd.concat([correct, correcti])

        with open(f"./results/logs/"+size+"_"+MDPid+".pkl","wb") as file:
            pickle.dump([df, correct, minGap], file)
        #print(df)
        #print(correct)
            