import logging
import numpy as np
from rlberry.agents import Agent
from utils import backward_induction_sd_active, policy_evaluation
logger = logging.getLogger(__name__)


class BPI_UCRLAgent(Agent):

    name = "BPI_UCRL"

    def __init__(
        self,
        env,
        delta=0.1, 
        epsilon=5,
        stage_dependent=True,
        period  = None,
        **kwargs
    ):
        Agent.__init__(self, env, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.stage_dependent = stage_dependent

       
        if period is None:
            self.period = self.env.S*self.env.A
        else:
            self.period = period
        
        
        # set of active actions
        # In case the MDP is not communicating, we only look at state-action pairs that are reachable 
        # starting from the initial state 
        self.Active = [[set() for ss in range(self.env.S)] for hh in range(self.env.horizon)] 
        self.Reachable = self.env.get_Reachable()
        for hh in range(self.env.horizon):
            for ss in self.Reachable[hh]:
                self.Active[hh][ss] = set(range(self.env.A))
        self.reset()

    def reset(self, **kwargs):
        del kwargs
        H = self.env.horizon
        S = self.env.S
        A = self.env.A

        # initial state (for stopping rule)
        self.initial_state = self.env.reset()
        
        # Boolean updated by stopping rules
        self.stop = False
        # Episode counter
        self.episode = 0

        if self.stage_dependent:
            shape_hs = (H, S)
            shape_hsa = (H, S, A)
            shape_hsas = (H, S, A, S)
        else:
            shape_hsa = (S, A)
            shape_hsas = (S, A, S)

        # D_h^t(s, a), used to compute the max diameter, used only for comparison with EPRL
        self.D_hsa = np.ones(shape_hsa)
        self.DV_hs = np.zeros(shape_hs)  # auxiliary only
        
        #upper bound on the optimal action-value function
        self.UQ_hsa = np.zeros(shape_hsa) 
        self.UV_hs = np.zeros(shape_hs)  # auxiliary only
        
        #lower bound on the optimal action-value function, u
        self.LQ_hsa = np.zeros(shape_hsa) 
        self.LV_hs = np.zeros(shape_hs)  # auxiliary only
        
    
        # N_h^t(s, a) and N_h^t(s,a,s'), counting the number of visits
        self.N_hsa = np.zeros(shape_hsa)
        self.N_hsas = np.zeros(shape_hsas)
        self.N_min = 0 
        # counting cumulative reward
        self.cumR = np.zeros(shape_hsa)
        self.bonus =  np.ones(shape_hsa)
        self.R_hat = np.zeros(shape_hsa)
        self.P_hat = self.env.P #np.ones(shape_hsas) * 1.0 / S

        #The true value functions, only used to evaluate correctness of the algorithm
        self.Qstar = np.zeros(shape_hsa)
        self.Vstar = np.zeros(shape_hs)
        backward_induction_sd_active(self.Qstar, self.Vstar, self.env.R, self.env.P, Active = self.Active)
            
    # Threshold for building confidence intervals on Q-functions
    def _beta(self, n):
        """
        Trying with empirical threshold (not supported by theory)
        """
        H = self.env.horizon
        S = self.env.S
        A = self.env.A
        delta = self.delta
        beta = 0.5* np.log((n + 1) / delta)  #0.5* np.log(S * A * H * np.exp(1) * (n + 1) / delta) 
        return beta

    # first stopping rule of EPRL: check if max_diameter < epsilon
    def get_max_diameter(self):
        maxD = -np.inf
        for aa in self.Active[0][self.initial_state]:
            if self.D_hsa[0, self.initial_state, aa] > maxD:
                maxD = self.D_hsa[0, self.initial_state, aa]
        return maxD
    # The "Q^star diameter" used in the stopping rule of BPI_UCRL, always smaller than the diameter D used in EPRL 
    def get_max_Qstar_diameter(self):
        maxU = np.max(self.UQ_hsa[0, self.initial_state, :])
        maxL = np.max(self.LQ_hsa[0, self.initial_state, :])
        return maxU - maxL
    
    # Computes average number of active actions per stage-state pairs (h,s)
    def get_avg_active(self):
        H = self.env.horizon
        Card = 0
        Count = 0
        for hh in range(H):
            for ss in self.Reachable[hh]:
                Count += 1
                Card += len(self.Active[hh][ss])
        return Card/Count
    
    # second stopping rule of EPRL, used for comparison of sampling rules of both algorithms
    def one_action_left(self): 
        for hh in range(self.env.horizon):
            for ss in self.Reachable[hh]:
                if len(self.Active[hh][ss]) > 1:
                    return False
        return True

    # Stopping rules of EPRL, used for comparison of sampling rules of both algorithms
    def EPRLstopping(self):
        self.stop = (self.get_max_diameter() < self.epsilon ) or \
                 self.one_action_left()
    
    def stopping(self):
        self.stop = (self.get_max_Qstar_diameter() < self.epsilon )
    
    #Recommendation rule : policy output by the algorithm
    def recommendation(self):
        return np.argmax(self.LQ_hsa, axis = 2)
        
    # Sampling rule : greedy policy w.r.t upper bound on Q^\star
    def exploration_policy(self):
        H = self.env.horizon
        S = self.env.S
        A = self.env.A
        pi = np.zeros((H,S), dtype = int)
        for hh in range(H):
            for ss in self.Reachable[hh]:
                maxUQ = -np.inf
                argmax = []
                for aa in self.Active[hh][ss]:
                    if self.UQ_hsa[hh, ss, aa] >= maxUQ:
                        if len(argmax) == 0:
                            argmax.append(aa)
                        else: # if there are multiple maximizers, we break ties in favor of the least played action
                            bb = argmax[0]
                            if self.UQ_hsa[hh, ss, aa] > maxUQ or self.N_hsa[hh, ss, aa] < self.N_hsa[hh, ss, bb]:
                                argmax[0] = aa
                        maxUQ = self.UQ_hsa[hh, ss, aa]
                pi[hh, ss] = argmax[0]
        return pi

        
    #Check if we need to perfom elimination rule. 
    #To reduce computational time, this done every once in a while
    def run_stopping(self):
        return self.episode%self.period == 0
        
       
    # Updating counts, reward estimates and bonuses     
    def _update(self, hh, state, action, reward, next_state):
        if self.stage_dependent:
            self.N_hsa[hh, state, action] += 1
            self.cumR[hh, state, action] += reward
            if hh < self.env.horizon -1:
                self.N_hsas[hh, state, action, next_state] += 1

            n_hsa = self.N_hsa[hh, state, action]
            #n_hsas = self.N_hsas[hh, state, action, :]
            self.R_hat[hh, state, action] = self.cumR[hh, state, action] / n_hsa
            #self.P_hat[hh, state, action, :] = n_hsas / n_hsa
            self.bonus[hh, state, action] = min( np.sqrt( self._beta(n_hsa) / n_hsa ), 1 )
   
            prev_N_min = self.N_min
            #we only care about the number of visits to state-action pairs that reachabe from s_0
            m = np.inf
            for hh in range(self.env.horizon):
                for ss in self.Reachable[hh]:
                    for aa in self.Active[hh][ss]:
                        if self.N_hsa[hh, ss, aa] < m:
                            m = self.N_hsa[hh, ss, aa]
            self.N_min = m
            
        else:
            self.N_hsa[hh, state, action] += 1
            self.N_hsas[hh, state, action, next_state] += 1
            self.cumR[hh, state, action] += reward
            n_hsa = self.N_hsa[hh, state, action]
            n_hsas = self.N_hsas[hh, state, action, :]
            
            self.R_hat[hh, state, action] = self.cumR[hh, state, action] / n_hsa
            self.P_hat[hh, state, action, :] = n_hsas / n_hsa

            self.bonus[state, action] = (
                 min( np.sqrt( self._beta(n_hsa) / n_hsa ), 1 )
                )
    def run_episode(self, check=False):
        state = self.env.reset()
        for hh in range(self.env.horizon):
            action = self.exploration_policy()[hh, state]
            next_s, reward, done, _ = self.env.step(action)
            del done
            #if hh == 1 :
                #print(f"hh {hh}, ss {state}, aa {action}")
                #print(f"exploration policy {self.exploration_policy()}")

            #self.counter.update(hh, state, action, next_s, 0.0)
            self._update(hh, state, action, reward, next_s)

            state = next_s

        
        
        if self.stage_dependent:
            H = self.env.horizon
            S = self.env.S
            # update Diameter
            backward_induction_sd_active(
                self.D_hsa,
                self.DV_hs,
                2*self.bonus.copy(),
                self.P_hat,
                self.Active,
                vmax=np.inf,
            )
            # update upper and lower bounds on Q^\star
            Ur = self.R_hat + self.bonus
            Lr = self.R_hat - self.bonus
            if self.N_min == 0:
                # when N_min = 0, R_hat is biased and 
                # the algorithm may never visit the (s,a) such that N_{s,a} = 0, since it is optimistic w.r.t UQ_hsa
                Ur = self.bonus.copy() 
                Lr = -self.bonus.copy()
            backward_induction_sd_active(
                self.UQ_hsa,
                self.UV_hs,
                Ur,
                self.P_hat,
                self.Active,
                vmax=np.inf,
            )

            backward_induction_sd_active(
                self.LQ_hsa,
                self.LV_hs,
                Lr,
                self.P_hat,
                self.Active,
                vmax=np.inf,
            )
            #if (self.LQ_hsa > self.Qstar).any() or (self.UQ_hsa < self.Qstar).any():
                #print("WARNIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIiIIIIING")
                #print( f" LQ = {np.where(self.LQ_hsa - self.Qstar > 0)}")
                #print( f" Qstar = {self.Qstar}")
                #print( f" UQ = {np.where(self.UQ_hsa - self.Qstar < 0)}")
        else:
            self.D_hsa, _ = backward_induction(
                2*self.bonus, self.P_hat, self.env.horizon, self.Active, vmax=np.inf
            )
        if check and self.N_min > 0:
            self.stopping()
            
            
        # write info
        if self.writer is not None:
            self.writer.add_scalar(
                "max_diameter", self.get_max_diameter(), self.episode
            )
            self.writer.add_scalar(
                "Qstar_diameter", self.get_max_Qstar_diameter(), self.episode
            )
            self.writer.add_scalar(
                "avg_actions", self.get_avg_active(), self.episode
            )
            
        self.episode += 1
        
    def eval(self, **kwargs):
        del kwargs
        H = self.env.horizon
        S = self.env.S
        A = self.env.A
        pi_hat = self.recommendation()
        Vpi = policy_evaluation(self.env, pi_hat)[0, self.initial_state]
        Vstar = self.Vstar[0, self.initial_state]
        return int(Vpi > Vstar - self.epsilon)
        
    def fit(self, budget, **kwargs):
        del kwargs
        while (not self.stop) and self.episode < budget:
            if self.run_stopping() :
                self.run_episode(check = True)
            else:
                self.run_episode()
            #print(f"Episode n°{self.episode}, N_min = {self.N_min}\
            #Max Qstar diameter = {self.get_max_Qstar_diameter()}, Max diameter = {self.get_max_diameter()},\
            #Average active actions = {self.get_avg_active()}")
            #print(f"N = {self.N_hsa}")
            #print(f" UQ {self.UQ_hsa}")
        pi_hat = self.recommendation()
        self.writer.add_scalar(
                "correct", self.eval(), self.episode
            )
        return self.episode, pi_hat