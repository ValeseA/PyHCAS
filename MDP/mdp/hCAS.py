import numpy as np
from .constants import *

class HCAS_MDP:
    def __init__(self):
        self.ranges = np.array(RANGES)
        self.thetas = np.array(THETAS)
        self.psis = np.array(PSIS)
        self.vowns = np.array(OWNSPEEDS)
        self.vints = np.array(INTRSPEEDS)
        self.pras = np.array(ACTIONS)
        self.discount_factor = discount_f
        self.turns = turns
        self.currentTau = 0.0

# Action functions for HCAS_MDP

def actionindex(mdp, a):
    return a + 1

def actions(mdp):
    return ACTIONS

def discount(mdp):
    return mdp.discount_factor

def n_actions(mdp):
    return len(ACTIONS)

def convert_s(v):
    # Converts stateType to a vector (same format as in Julia)
    return [v[0], v[1], v[2], v[3], v[4], float(v[5])]

def convert_s_from_vector(v):
    # Converts vector back to stateType (tuple)
    return (v[0], v[1], v[2], v[3], v[4], int(v[5]))
