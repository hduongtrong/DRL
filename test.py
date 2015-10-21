from rl import rollout, Policy
from frozen_lake import FrozenLake
import numpy as np, numpy.random as nr
import ipdb

np.random.seed(1)

map4x4 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"]
mdp = FrozenLake(map4x4)

# FrozenLake is a MDP with finite state and action that involves navigating across a frozen lake.
# (It's conventionally called a "grid-world" MDP, as the state space involves points in a 2D grid)
# Let's look at the docstring for details
print FrozenLake.__doc__
print "-----------------"

class RandomDiscreteActionChooser(Policy):
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def step(self, observation):
        return {"action":np.array([nr.randint(0, self.n_actions)])}

policy = RandomDiscreteActionChooser(mdp.n_actions)
    
rdata = rollout(mdp, policy, 100)
print rdata

s_n = rdata['observations'] # Vector of states (same as observations since MDP is fully-observed)
a_n = rdata['actions'] # Vector of actions (each is an int in {0,1,2,3})
n = a_n.shape[0] # Length of trajectory
q_n = np.random.randn(n) # Returns (random for the sake of gradient checking)
f_sa = np.random.randn(mdp.n_states, mdp.n_actions) # Policy parameter vector. explained shortly.

def softmax_prob(f_na):    
    """
    Exponentiate f_na and normalize rows to have sum 1
    so each row gives a probability distribution over discrete
    action set
    """
    prob_nk = np.exp(f_na - f_na.max(axis=1,keepdims=True))
    prob_nk /= prob_nk.sum(axis=1,keepdims=True)
    return prob_nk

def softmax_policy_checkfunc(f_sa, s_n, a_n, q_n):
    r"""
    An auxilliary function that's useful for checking the policy gradient
    The inputs are 
    s_n : states (vector of int)
    a_n : actions (vector of int)
    q_n : returns (vectof of float)

    This function returns

    \sum_n \log \pi(a_n | s_n) q_n

    whose gradient is the policy gradient estimator

    \sum_n \grad \log \pi(a_n | s_n) q_n

    """
    f_na = f_sa[s_n]
    p_na = softmax_prob(f_na)
    n = s_n.shape[0]
    return np.log(p_na[np.arange(n), a_n]).dot(q_n)/n

def softmax_policy_gradient(f_sa, s_n, a_n, adv_n):
    """
    Compute policy gradient of policy for discrete MDP, where probabilities
    are obtained by exponentiating f_sa and normalizing
    """
    # YOUR CODE HERE >>>>>>
    # <<<<<<<< 
    f_na = f_sa[s_n]
    p_na = softmax_prob(f_na)
    n = s_n.shape[0]
    grad_sa = np.zeros(f_sa.shape)
    for i in xrange(n):
        this_grad_sa = np.zeros(f_sa.shape)
        this_grad_sa[s_n[i], :] = -p_na[i]
        this_grad_sa[s_n[i], a_n[i]] += 1
        this_grad_sa = this_grad_sa * adv_n[i]
        grad_sa += this_grad_sa
    
    return grad_sa / n
    

from hw_utils import Message, discount, fmt_row
from rl import rollout, pathlength
import numpy as np
from collections import defaultdict
from categorical import cat_sample, cat_entropy, cat_kl
import matplotlib.pyplot as plt

class FrozenLakeTabularPolicy(Policy):
    def __init__(self, n_states):
        self.n_states = n_states
        self.n_actions = n_actions = 4        
        self.f_sa = np.zeros((n_states, n_actions))

    def step(self, s_n):
        f_na = self.f_sa[s_n]
        prob_nk = softmax_prob(f_na)
        acts_n = cat_sample(prob_nk)
        return {"action":acts_n,
                "pdist" : f_na}
    
    
    def compute_pdists(self, s_n):
        return self.f_sa[s_n]

    def compute_entropy(self, f_na):
        prob_nk = softmax_prob(f_na)
        return cat_entropy(prob_nk)

    def compute_kl(self, f0_na, f1_na):
        p0_na = softmax_prob(f0_na)
        p1_na = softmax_prob(f1_na)
        return cat_kl(p0_na, p1_na)

def policy_gradient_optimize_nesterov(mdp, policy,
        gamma,
        max_pathlength,
        timesteps_per_batch,
        n_iter,
        stepsize,
        beta = .95):
    stat2timeseries = defaultdict(list)
    widths = (17,10,10,10,10)
    print fmt_row(widths, ["EpRewMean","EpLenMean","Perplexity","KLOldNew"])
    
    fprev_sa = policy.f_sa
    for i in xrange(n_iter):
        total_ts = 0
        paths = [] 
        while True:
            path = rollout(mdp, policy, max_pathlength)                
            paths.append(path)
            total_ts += pathlength(path)
            if total_ts > timesteps_per_batch: 
                break
        
        # get observations:
        obs_no = np.concatenate([path["observations"] for path in paths])
        z_sa = policy.f_sa + beta * (policy.f_sa - fprev_sa) # Momemtum term
        grad = 0
        for path in paths:
            n = len(path['rewards'])
            q_n = ((path['rewards'] * gamma ** np.arange(n) )[::-1].cumsum())[::-1]
            q_n = q_n / gamma ** np.arange(n) # Biased estimator but doesn't decay as fast
            grad += softmax_policy_gradient(z_sa, path['observations'], 
                                            path['actions'], q_n)
        grad = grad / len(paths)
        fprev_sa = policy.f_sa
        policy.f_sa = z_sa + stepsize * grad
        
        pdists = np.concatenate([path["pdists"] for path in paths])
        kl = policy.compute_kl(pdists, policy.compute_pdists(obs_no)).mean()
        perplexity = np.exp(policy.compute_entropy(pdists).mean())

        stats = {  "EpRewMean" : np.mean([path["rewards"].sum() for path in paths]),
                   "EpRewSEM" : np.std([path["rewards"].sum() for path in paths])/np.sqrt(len(paths)),
                   "EpLenMean" : np.mean([pathlength(path) for path in paths]),
                   "Perplexity" : perplexity,
                   "KLOldNew" : kl }
        print fmt_row(widths, ['%.3f+-%.3f'%(stats["EpRewMean"], stats['EpRewSEM']), stats['EpLenMean'], stats['Perplexity'], stats['KLOldNew']])
        
        
        for (name,val) in stats.items():
            stat2timeseries[name].append(val)
    return stat2timeseries

map8x8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
]
mdp8 = FrozenLake(map8x8)
policy = FrozenLakeTabularPolicy(mdp8.n_states)

np.random.seed(0)
stat2ts = policy_gradient_optimize_nesterov(mdp8, policy,
                gamma=.9,
                max_pathlength=400,
                timesteps_per_batch=8000,
                n_iter=200,
                stepsize=1000)
