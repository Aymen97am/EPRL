import itertools
import numpy as np
from numpy.random import default_rng
from utils import backward_induction_sd_active, policy_evaluation
import logging

import rlberry.spaces as spaces
from rlberry.envs.interface import Model

logger = logging.getLogger(__name__)


class FiniteDMDP(Model):
    """
    Base class for a finite episodic MDP with deterministic  transitions.

    Terminal states are set to be absorbing, and
    are determined by the is_terminal() method,
    which can be overriden (and returns false by default).

    Parameters
    ----------
    R : numpy.ndarray
    P : numpy.ndarray
    initial_state_distribution : numpy.ndarray or int
        array of size (S,) containing the initial state distribution
        or an integer representing the initial/default state
    sigma : float containing standard-deviation of the gaussian rewards.

    Attributes
    ----------
    R : numpy.ndarray
        array of shape (H, S, A) containing the mean rewards, where
        H = Horizon, S = number of states;  A = number of actions.
    P : numpy.ndarray
        array of shape (H-1, S, A, S) containing the transition probabilities,
        where P[h, s, a, s'] = Prob(S_{h+1}=s'| S_h = s, A_h = a).
    """

    def __init__(self, R, P, initial_state_distribution=0, sigma = 0.5):
        Model.__init__(self)
        
        self._rng = default_rng() #Random number generator that will be used for the MDP
        self.initial_state_distribution = initial_state_distribution
        H, S, A = R.shape

        self.horizon = H
        self.S = S
        self.A = A
       
        self.R = R
        self.P = P
        self.sigma = sigma # standard-deviation for the rewards

        
        self.horizon_space = spaces.Discrete(H)
        self.observation_space = spaces.Discrete(S)
        self.action_space = spaces.Discrete(A)
        self.reward_range = (self.R.min(), self.R.max())

        self.stage = 0
        self.state = None

        self._states = np.arange(S)
        self._actions = np.arange(A)

        self.reset()
        self._check()

    def reset(self):
        """
        Reset the environment to a default state.
        """
        if isinstance(self.initial_state_distribution, np.ndarray):
            self.state = self.rng.choice(
                self._states, p=self.initial_state_distribution
            )
        else:
            self.state = self.initial_state_distribution
            self.stage = 0
        self._rng = default_rng() #Reinitialize the random number generator 
        return self.state


    def _check_init_distribution(self):
        if isinstance(self.initial_state_distribution, np.ndarray):
            assert abs(self.initial_state_distribution.sum() - 1.0) < 1e-15
        else:
            assert self.initial_state_distribution >= 0
            assert self.initial_state_distribution < self.S

    def _check(self):
        """
        Check consistency of the MDP
        """
        # Check initial_state_distribution
        self._check_init_distribution()

        # Check that P[s,a, :] is a probability distribution
        for h in range(self.horizon - 1):
            for s in self._states:
                for a in self._actions:
                    assert abs(self.P[h, s, a, :].sum() - 1.0) < 1e-15

        # Check that dimensions match
        H1, S1, A1 = self.R.shape
        H2, S2, A2, S3 = self.P.shape
        assert H1 == H2 + 1
        assert S1 == S2 == S3
        assert A1 == A2

    def get_Reachable(self):
        """
        Get the set of reachable states at every layer
        """
        Reachable = [set({self.initial_state_distribution})]
        for h in range(self.horizon-1):
            Reachable.append({})
            temp = set()
            for s in Reachable[h]:
                for a in range(self.A):
                    Next_s = np.where(self.P[h, s, a] == 1)[0][0]
                    temp.add(Next_s)
            Reachable[h+1] = temp
        return Reachable
    def set_initial_state_distribution(self, distribution):
        """
        Parameters
        ----------
        distribution : numpy.ndarray or int
            array of size (S,) containing the initial state distribution
            or an integer representing the initial/default state
        """
        self.initial_state_distribution = distribution
        self._check_init_distribution()

    def sample(self, state, action):
        """
        Sample a transition s' from P_stage(s'|state, action) 
        and a reward from gaussian with mean R[h,s,a] and variance self.sigma^2.
        """
        stage = self.stage
        next_state = None
        if self.stage < self.horizon -1: 
            prob = self.P[stage, state, action, :]
            next_state = self.rng.choice(self._states, p=prob)
        reward = self.reward_fn(stage, state, action, next_state) + self.sigma*self._rng.standard_normal()
        info = {}
        return next_state, reward, info

    def step(self, action):
        assert action in self._actions, "Invalid action!"
        next_state, reward, info = self.sample(self.state, action)
        self.state = next_state
        self.stage+= 1
        self.stage = min(self.horizon-1, self.stage)
        done = self.is_terminal()
        return next_state, reward, done, info

    def is_terminal(self):
        """
        Returns true if a state is terminal.
        """
        if self.stage == self.horizon:
            return True
        return False

    def reward_fn(self, stage, state, action, next_state):
        """
        Reward function. Returns mean reward at (state, action)

        Parameters
        ----------
        state : int
            current state
        action : int
            current action
        next_state :
            next state

        Returns:
            reward : float
        """
        return self.R[stage, state, action]

    def log(self):
        """
        Print the structure of the MDP.
        """
        indent = "    "
        for h in range(self.horizon):
            logger.info(f"Stage {h} {indent}")
            for s in self._states:
                logger.info(f"State {s} {indent}")
                for a in self._actions:
                    logger.info(f"{indent} Action {a}")
                    for ss in self._states:
                        if self.P[h, s, a, ss] > 0.0:
                            logger.info(
                                f"{2 * indent} transition to {ss} "
                                f"with prob {self.P[h, s, a, ss]: .2f}"
                            )
                logger.info("~~~~~~~~~~~~~~~~~~~~")

    
    def Complexity(self):
        """
        Returns  (complexity1/complexity2, minGap) where
        complexity1 : sum_{h,s,a} 1/value_gap[h,s,a]^2
        complexity2 : sum_{h,s,a} 1/return_gap[h,s,a]^2
        minGap : Minimum value gap in the MDP
        """
        H = self.horizon
        S = self.S
        A = self.A
        Qstar = np.zeros((H, S, A))
        Vstar = np.zeros((H, S))
        backward_induction_sd_active(Qstar, Vstar, self.R, self.P, Active = None)
        Vstar = np.repeat(Vstar, A)
        Vstar = Vstar.reshape((H, S, A))
        ValueGaps = Vstar - Qstar
        ValueGaps = ValueGaps[np.where(ValueGaps > 0)]
        ValueGapsComplexity = np.sum(1/ValueGaps**2)
        minGap = np.min(ValueGaps)
        ReturnGaps = np.zeros((H, S, A))
        Reachable = self.get_Reachable()
        for hh in range(H):
            for ss in Reachable[hh]:
                for aa in range(A):
                    Ur_hsa =  self.R.copy() #upper bounds on the reward
                    temp = Ur_hsa[hh, ss, aa]
                    Ur_hsa[hh,:, :] = -np.inf #penalize policies that don't pass through (h,s,a)
                    Ur_hsa[hh, ss, aa] = temp
                    UQ = np.zeros((H, S, A))
                    UV = np.zeros((H, S))
                    backward_induction_sd_active(
                        UQ,
                        UV,
                        Ur_hsa,
                        self.P,
                        None,
                        vmax = np.inf,
                    )
                    ReturnGaps[hh, ss, aa] = Vstar[0, self.initial_state_distribution, 0] -\
                    UV[0, self.initial_state_distribution] 
                    
        ReturnGaps = ReturnGaps[np.nonzero(ReturnGaps)]
        ReturnGapsComplexity = np.sum(1/ReturnGaps**2)
        return ValueGapsComplexity/ReturnGapsComplexity, minGap
    
    @classmethod
    def Random(cls, H, S, A, Rmax = 1, initial_state_distr=0, sigma = 0.5): 
        """
        Generates a random MDP with deterministic transitions
        """
        R = np.random.uniform(0, Rmax, (H, S, A))
        NEXT = np.random.choice(S, (H-1, S, A))
        P = np.zeros((H-1, S, A, S))
        for h, s, a in itertools.product(range(H-1),range(S),range(A)):
            next_s = NEXT[h, s, a]
            P[h, s, a, next_s] = 1
        return FiniteDMDP(R, P, initial_state_distribution=initial_state_distr, sigma = sigma)
    
    @classmethod
    def RandomwithThreshold(cls, H, S, A, threshold = 100, Rmax = 1, initial_state_distr=0, sigma = 0.5):
        """
        Generates a random MDP with deterministic transitions whose ratio of 
        ValueGapsComplexity/ReturnGapsComplexity is in [threshold, 2*threshold]
        
        """
    
        env = cls.Random(H, S, A, Rmax = 1, initial_state_distr=0, sigma = sigma)
        ratio, _ = env.Complexity()
        while ratio < threshold or ratio > 2*threshold:
            env = cls.Random(H, S, A, Rmax = 1, initial_state_distr=0, sigma = sigma)
            ratio, _ = env.Complexity()
        print(f"Ratio of complexities = {ratio}")
        return env
    
    @classmethod
    def RandomwithGapThreshold(cls, H, S, A, threshold = 0.1, Rmax = 1, initial_state_distr=0, sigma = 0.5):
        """
        Generates a random MDP with deterministic transitions whose minimum value gap is 
        larger than threshold
        
        """
    
        env = cls.Random(H, S, A, Rmax = 1, initial_state_distr=0, sigma = sigma)
        _, minGap = env.Complexity()
        count = 0
        while minGap < threshold and count < 1e2:
            env = cls.Random(H, S, A, Rmax = 1, initial_state_distr=0, sigma = sigma)
            _, minGap = env.Complexity()
        if count == 1e2:
            raise ValueError("The threshold that you have set for the minimum gap is too large\
            please set a small threshold")
        return env, minGap
          
            
    
    
