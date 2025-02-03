import numpy as np
import h5py
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from tqdm import tqdm  # For better progress tracking

from mdp.constants import *
from mdp.hCAS import *
from mdp.rewards import *
from mdp.transitions import *
from GridInterpolations.gridInterpolations import *
from LocalFunctionApproximation.localFunctionApproximator import *

# Parameters
saveFile = "./Qtables/HCAS_oneSpeed_v6_diff_dists.h5"  # Output file for Q-tables
nTau_warm = 99  # Number of warm-up iterations
nTau_max = 60  # Maximum value for tau

# Create MDP instances for each value of tau
mdps = [HCAS_MDP() for _ in range(nTau_max + 1)]
for tau, mdp in enumerate(mdps):
    mdp.currentTau = tau

def weighted_iterator(dist):
    """
    Iterator for weighted states and probabilities.

    Args:
        dist (tuple): A tuple containing states and their probabilities.

    Yields:
        Tuple of (state, probability).
    """
    states, probabilities = dist
    for state, probability in zip(states, probabilities):
        yield state, probability

def compute_helper(states, n_states, mdps, grid, a, nTau_max):
    """
    Compute the transition matrix and rewards for a given action.

    Args:
        states (list): List of all states.
        n_states (int): Number of states.
        mdps (list): List of MDP instances for different tau values.
        grid (Grid): Grid used for state interpolation.
        a (int): Action index.
        nTau_max (int): Maximum tau value.

    Returns:
        tuple: Transition matrix (CSR) and rewards array.
    """
    rval = np.zeros(n_states * 100, dtype=np.int32)
    cval = np.zeros(n_states * 100, dtype=np.int32)
    zval = np.zeros(n_states * 100, dtype=np.float64)
    rews = [np.zeros(n_states,dtype=np.float64) for _ in range(nTau_max + 1)]
    index = 0
    fracBase = 0.2  # Fractional progress intervals for logging
    frac = fracBase

    # Iterate over all states
    for i, s in enumerate(tqdm(states, desc="State processing")):
        #if i / n_states >= frac:
        #    print(f"{round(frac * 100)}% Complete")
        #    frac += fracBase

        # Debug: Print rewards for the first few states
        #if i < 5:
        #    print(f"Rewards for state {i}: {rews[0][i]}")

        # Compute rewards for all tau values
        for tau in range(nTau_max + 1):
            rews[tau][i] = reward(mdps[tau], s, a)

        # Compute transitions for the first tau (debug)
        dist = transition(mdps[0], s, a)
        for sp, p in weighted_iterator(dist):
            if p > 0.0:
                sp_point = convert_s(sp)
                sps, probs = interpolants(grid, sp_point)
                for spi, probi in zip(sps, probs):
                    rval[index] = i
                    cval[index] = spi
                    zval[index] = probi * p
                    index += 1

        # Debug: Print transition data for the first few states
        #if i < 5:
        #    print(f"Transitions for state {i}: rval={rval[:index]}, cval={cval[:index]}, zval={zval[:index]}")

    # Create a sparse transition matrix
    trans = csr_matrix((zval[:index], (rval[:index], cval[:index])), shape=(n_states, n_states))
    return trans, rews

def compute_Qa(r, gam, trans, U):
    """
    Compute the Q-value for a given action.

    Args:
        r (array): Rewards for the action.
        gam (float): Discount factor.
        trans (csr_matrix): Transition matrix for the action.
        U (array): Utility values from the previous iteration.

    Returns:
        array: Q-values for the action.
    """
    return r + gam * trans.dot(U)

def compute_trans_reward(mdps, interp, nTau_max):
    """
    Compute transition matrices and rewards for all actions.

    Args:
        mdps (list): List of MDP instances.
        interp (Interpolator): Interpolator for states.
        nTau_max (int): Maximum tau value.

    Returns:
        tuple: Transition matrices and rewards for all actions.
    """
    t, rews = {}, {}
    n_states = len(interp.grid)
    print(f"Total states: {n_states}")
    interp_points = interp.get_all_interpolating_points()
    interp_states = [convert_s(pt) for pt in interp_points]

    def parallel_compute_helper(ai, a):
        return compute_helper(interp_states, n_states, mdps, interp.grid, a, nTau_max)

    # Parallel computation for each action
    results = Parallel(n_jobs=-1)(
        delayed(parallel_compute_helper)(ai, a) for ai, a in enumerate(actions(mdps[0]))
    )

    for ai, result in enumerate(results):
        t[ai], rews[ai] = result

    return t, rews

def computeQ(mdps, interp, nTau_warm, nTau_max):
    """
    Compute Q-values for the given MDPs and interpolation parameters.

    Args:
        mdps (list): List of MDP instances.
        interp (Interpolator): State interpolation object.
        nTau_warm (int): Number of warm-up iterations.
        nTau_max (int): Maximum tau value.

    Returns:
        array: Q-values for all states and actions.
    """
    trans, rews = compute_trans_reward(mdps, interp, nTau_max)
    ns = len(rews[0][0])  # Number of states
    na = len(actions(mdps[0]))  # Number of actions
    nt = nTau_max + 1  # Total number of tau values
    gam = discount(mdps[0])  # Discount factor
    U = np.zeros(ns,dtype=np.float64)  # Utility array
    Q = np.zeros((ns, na),dtype=np.float64)  # Q-value array
    Q_out = np.zeros((ns * nt, na),dtype=np.float64)  # Output Q-values

    # Warm-up phase for tau=0
    print("Starting warm-up phase...")
    for i in tqdm(range(nTau_warm)):
        #print(f"Warm-up iteration {i + 1}/{nTau_warm}")
        results = Parallel(n_jobs=-1)(
            delayed(compute_Qa)(rews[ai][0], gam, trans[ai], U) for ai in range(na)
        )
        Q = np.column_stack(results)
        U = np.max(Q, axis=1)  # Update utility values

        #if (i + 1) % 10 == 0 or i == 0 or i == nTau_warm - 1:
        #    print(f"Warm-up Q values (iteration {i + 1}):\n{Q[:5, :]}")

    # Main computation for tau = 1 to nTau_max
    print("Starting main computation phase...")
    for i in tqdm(range(nt)):
        #print(f"Computing Q values for tau={i} ({i + 1}/{nt})")
        results = Parallel(n_jobs=-1)(
            delayed(compute_Qa)(rews[ai][i], gam, trans[ai], U) for ai in range(na)
        )
        Q = np.column_stack(results)
        U = np.max(Q, axis=1)
        Q_out[i * ns:(i + 1) * ns, :] = Q.copy()

        #if (i + 1) % 10 == 0 or i == 0 or i == nt - 1:
        #    print(f"Q values for tau={i}:\n{Q[:5, :]}")

    print("Q computation completed.")
    return Q_out

# Define the grid and the function approximator
grid = RectangleGrid(RANGES, THETAS, PSIS, OWNSPEEDS, INTRSPEEDS, ACTIONS)
interp = LocalGIFunctionApproximator(grid)

# Compute Q-values
Q_out = computeQ(mdps, interp, nTau_warm, nTau_max).T
print(type(Q_out), Q_out.shape)

# Save results
print("Saving Q-values...")
with h5py.File(saveFile, "w") as H:
    H.create_dataset("q", data=Q_out)
    H.create_dataset("ranges", data=RANGES)
    H.create_dataset("thetas", data=THETAS)
    H.create_dataset("psis", data=PSIS)
