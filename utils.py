import numpy as np

def policy_evaluation(env, pi):
    H, S, A = env.R.shape
    R = env.R
    P = env.P
    assert pi.shape == (H,S), "Policy array has wrong shape !"
    Q = np.zeros((H, S, A))
    V = np.zeros((H, S))
    for hh in range(H - 1, -1, -1):
        for ss in range(S):
            for aa in range(A):
                Q[hh, ss, aa] = R[hh, ss, aa]
                if hh < H - 1:
                    for ns in range(S):
                        Q[hh, ss, aa] += P[hh, ss, aa, ns] * V[hh + 1, ns]
            a_pi = pi[hh, ss]
            V[hh, ss] = Q[hh, ss, a_pi]

    return V

def backward_induction_sd_active(Q, V, R, P, Active, vmax=np.inf): #induction only over active actions
    """
    In-place implementation of backward induction to compute Q and V functions
    in the finite horizon setting.
    Assumes R and P are stage-dependent.
    Parameters
    ----------
    Q:  numpy.ndarray
        array of shape (H, S, A) where to store the Q function
    V:  numpy.ndarray
        array of shape (H, S) where to store the V function
    R : numpy.ndarray
        array of shape (H, S, A) contaning the rewards, where S is the number
        of states and A is the number of actions
    P : numpy.ndarray
        array of shape (H, S, A, S) such that P[h, s, a, ns] is the probability of
        arriving at ns by taking action a in state s at stage h.

    vmax : double, default: np.inf
        maximum possible value in V
        
    """
    H, S, A = R.shape
    if Active is None:
        Active = [[set(range(A)) for ss in range(S)] for hh in range(H)]
    for hh in range(H - 1, -1, -1):
        for ss in range(S):
            max_q = -np.inf
            for aa in Active[hh][ss]:
                q_aa = R[hh, ss, aa]
                if hh < H - 1:
                    # not using .dot instead of loop to avoid scipy dependency
                    # (numba seems to require scipy for linear algebra
                    #  operations in numpy)
                    for ns in range(S):
                        p = P[hh, ss, aa, ns]
                        # To handle the case where p == 0 and V = -np.inf 
                        # (used in artificial rewards of the elimination rule)
                        if p > 0 : 
                            q_aa += p * V[hh + 1, ns]
                if q_aa > max_q:
                    max_q = q_aa
                Q[hh, ss, aa] = q_aa
            V[hh, ss] = max_q
            if V[hh, ss] > vmax:
                V[hh, ss] = vmax