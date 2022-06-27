from rlberry.manager import AgentManager, evaluate_agents
import rlberry.manager as manager
from rlberry.manager.evaluation import read_writer_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Put all agents results in a unified dataframe
def process_results(agents, tag="max_diameter"):
    global_df = pd.DataFrame()
    for agent in agents:
        df = read_writer_data(agent, tag=tag)
        df["name"] = df["name"].astype(str)
        for col in ["value", "global_step", "n_simu"]:
            df[col] = pd.to_numeric(df[col])
        global_df = pd.concat([global_df, df])
    global_df = global_df.rename(columns = {"value" : tag, "global_step" : "episode"})
    #global_df["name"] = global_df["name"].astype(str)
    return global_df
def plot(agents, names, env_kwargs, variable="tau", plot = False):
    #dictionary for values of interest to be returned
    log = {}
    if variable == "tau" or variable == "max_diameter":
        df = process_results(agents, tag= "max_diameter")
        df1 = process_results(agents, tag= "Qstar_diameter")
    else:
        df = process_results(agents, tag= variable) 

    n_sim = df["n_simu"].max()+1
    
    if variable == "correct":
        for agent in agents:
            name = names[str(agent)]
            Correct = []
            for sim in range(n_sim): #loop over montecarlo simulations
                # get the index of the last episode (=stopping time)
                tau = df[(df["name"] == name) & (df["n_simu"] == sim)]["episode"].max()
                #Check correctness of the policy recommended at the end
                correct = df[(df["name"] == name) & (df["n_simu"] == sim) & (df["episode"] == tau)]["value"]
                Correct.append(correct)
            log[name] = np.mean(Correct)
        return log
    elif variable == "tau":
        data = {}
        for agent in agents:
            name = names[str(agent)]
            Taus = []
            for sim in range(n_sim): #loop over montecarlo simulations
                # get the index of the last episode (=stopping time)
                tau = df[(df["name"] == name) & (df["n_simu"] == sim)]["episode"].max()
                Taus.append(tau)
            data[name] = Taus 
            log[name] = np.mean(Taus)
        fig, ax = plt.subplots()
        ax.set_title('Stopping time')
        ax.boxplot(data.values(), labels = data.keys(), whis = (1, 99))
    elif variable == "max_diameter":
        data = {}
        data1 = {}
        for agent in agents:
            name = names[str(agent)]
            Diams = []
            QstarDiams = []
            minL = np.inf
            for sim in range(n_sim):
                D = df[(df["name"] == name) & (df["n_simu"] == sim)]["value"]
                QstarD = df1[(df1["name"] == name) & (df1["n_simu"] == sim)]["value"]
                Diams.append(D)
                QstarDiams.append(QstarD)
                if len(D) < minL:
                    minL = len(D)
            for sim in range(n_sim):
                Diams[sim] = Diams[sim][:minL]
                QstarDiams[sim] = QstarDiams[sim][:minL]
            data[name] = np.mean(np.array(Diams), axis = 0) 
            data1[name] = np.mean(np.array(QstarDiams), axis = 0)
        Nagents = len(agents)
        fig, ax = plt.subplots(1,Nagents, figsize = (20,10))
        #ax.set_title('Diameters')
        for i, agent in enumerate(agents):
            name = names[str(agent)]
            minL = len(data[name])
            ax[i].set_title(name)
            ax[i].plot(np.arange(0,minL),data[name], label = "max_diameter")
            ax[i].plot(np.arange(0,minL),data1[name], label = "Qstar_diameter")
            plt.legend()
    elif variable == "avg_actions":
        data = {}
        for agent in agents:
            name = names[str(agent)]
            Nactions = []
            minL = np.inf
            for sim in range(n_sim):
                N = df[(df["name"] == name) & (df["n_simu"] == sim)]["value"] 
                Nactions.append(N)
                if len(N) < minL:
                    minL = len(N)
            for sim in range(n_sim):
                Nactions[sim] = Nactions[sim][:minL]
            data[name] = np.mean(np.array(Nactions), axis = 0) 
        Nagents = len(agents)
        fig, ax = plt.subplots(1,Nagents, figsize = (20,10))
        for i, agent in enumerate(agents):
            name = names[str(agent)]
            minL = len(data[name])
            ax[i].set_title(name)
            ax[i].plot(np.arange(0,minL),data[name], label = "Average number of active actions")
            plt.legend()
    S = env_kwargs["S"]
    A = env_kwargs["A"]
    H = env_kwargs["H"]
    epsilon = agents[0].get_agent_instances()[0].epsilon
    plt.savefig(f"./results/figures/{variable}_S{S}A{A}H{H}_epsilon{epsilon}.png")
    plt.show()
    return log