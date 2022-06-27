import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

"""
Reads .pkl files where experiment logs are saved and recovers stopping times
of each algorithm for different values of epsilon
"""

if __name__ == '__main__':

    path = "./results/logs/"
    dfs = []
    corrects = []
    minGaps = []
    for filename in os.listdir(path):
        if filename[-4:] == ".pkl":
            file = open(path+filename, 'rb')
            df, correct, minGap = pickle.load(file)
            file.close()
            dfs.append(df)
            corrects.append(correct)
            minGaps.append(minGap)
        
        
    N = len(minGaps)
    #order alphas decreasingly to speed up run time
    alphas = [10, 5, 2.5] + list(np.flip(np.arange(0.05, 2.5, 0.1))) # value of the ratio epsilon/Delta_min
    algos = ["BPI_UCRL", "maxD", "AmaxCov"]
    n_sim = 10 # number of monte-carlo simulationq
    DATA = pd.DataFrame()
    diameter = {"BPI_UCRL" : "Qstar_diameter", "maxD": "max_diameter", "AmaxCov": "max_diameter"}
    for n in range(N):
        df = dfs[n]
        MDP_name = df["MDP"].iloc[0]
        Taus = {}
        for name in algos:
            df_alg = df[(df["name"]== name)]
            Taus[name] = {}
            for alpha in alphas:
                Taus[name][alpha] = []
            for sim in range(n_sim):
                df_alg_sim = df_alg[df_alg["n_simu"]== sim]
                max_ep = df_alg_sim["episode"].max()
                for alpha in alphas:
                    epsilon = alpha*minGaps[n]
                    df_alg_sim = df_alg_sim[(df_alg_sim[diameter[name]] < epsilon)]
                    tau = df_alg_sim["episode"].min() #stopping time
                    # in case EPRL because only one active action remains in every (s,h)
                    # so the diameter never went below epsilon
                    if np.isnan(tau): 
                        tau = max_ep
                    #print(name, alpha, sim, tau)
                    Taus[name][alpha].append(tau)  
        for alpha in alphas:
            data = {"MDP": [], "alpha" : [], "minGap": [], "BPI_UCRL" : [], "maxD": [], "AmaxCov": []}
            data["MDP"].append(MDP_name)
            data["alpha"].append(alpha)
            data["minGap"].append(minGaps[n])
            for name in algos: 
                data[name].append(np.mean(Taus[name][alpha]))
            DATA = pd.concat([DATA, pd.DataFrame(data)])
    with open(f"./results/logs/processed/DATA.pkl","wb") as file:
            pickle.dump(DATA, file)