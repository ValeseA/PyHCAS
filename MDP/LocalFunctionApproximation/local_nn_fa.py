import numpy as np
from typing import List, Union, Optional

# Class definition
class LocalNNFunctionApproximator:
    def __init__(self, nntree, nnpoints: List[np.ndarray], knnK: Optional[int] = None, rnnR: Optional[float] = None):
        self.nntree = nntree
        self.nnpoints = nnpoints
        self.nnvalues = np.zeros(len(nntree.indices))
        self.knnK = knnK
        self.rnnR = rnnR

    @classmethod
    def from_knn(cls, nntree, nnpts: List[np.ndarray], knnK: int):
        return cls(nntree, nnpts, knnK=knnK)

    @classmethod
    def from_rnn(cls, nntree, nnpts: List[np.ndarray], rnnR: float):
        return cls(nntree, nnpts, rnnR=rnnR)

# Interface functions
def n_interpolating_points(nnfa: LocalNNFunctionApproximator) -> int:
    return len(nnfa.nntree.indices)

def get_all_interpolating_points(nnfa: LocalNNFunctionApproximator) -> List[np.ndarray]:
    return nnfa.nnpoints

def get_all_interpolating_values(nnfa: LocalNNFunctionApproximator) -> np.ndarray:
    return nnfa.nnvalues

def get_interpolating_nbrs_idxs_wts(nnfa: LocalNNFunctionApproximator, v: np.ndarray):
    assert nnfa.knnK is not None or nnfa.rnnR is not None, "Either knnK or rnnR must be defined"
    
    if nnfa.knnK is not None:
        # Perform k-NN lookup to get data and distances
        idxs, dists = knn(nnfa.nntree, v, nnfa.knnK)
    else:
        # Perform in-range lookup to get data
        idxs = inrange(nnfa.nntree, v, nnfa.rnnR)
        dists = np.zeros(len(idxs))
        for i, idx in enumerate(idxs):
            dists[i] = Distances.evaluate(nnfa.nntree.metric, v, nnfa.nnpoints[idx])

    weights = np.zeros(len(dists))

    # If exactly one point, set that probability to 1 and others to 0
    if np.min(dists) < np.finfo(np.float64).eps:
        weights[np.argmin(dists)] = 1.0
    else:
        for i, d in enumerate(dists):
            weights[i] = 1.0 / d
        weights /= np.sum(weights)

    return idxs, weights

def compute_value(nnfa: LocalNNFunctionApproximator, v: np.ndarray) -> float:
    idxs, wts = get_interpolating_nbrs_idxs_wts(nnfa, v)

    # Perform a weighted average of values
    value = 0.0
    wtsum = 0.0
    for i, idx in enumerate(idxs):
        value += wts[i] * nnfa.nnvalues[idx]
        wtsum += wts[i]
    value /= wtsum

    return value

def compute_value(nnfa: LocalNNFunctionApproximator, v_list: List[np.ndarray]) -> List[float]:
    assert len(v_list) > 0, "The input list must not be empty"
    vals = [compute_value(nnfa, pt) for pt in v_list]
    return vals

def set_all_interpolating_values(nnfa: LocalNNFunctionApproximator, nnvalues: np.ndarray):
    nnfa.nnvalues = np.copy(nnvalues)
