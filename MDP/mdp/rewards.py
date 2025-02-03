import numpy as np
from .constants import COC  # Ensure COC is defined in constants

# Supporting functions
def sameSense(pra, ra):
    """
    Check if pra (previous resolution advisory) and ra (current resolution advisory) 
    have the same parity, indicating the same general sense of direction.

    Args:
        pra (int): Previous resolution advisory.
        ra (int): Current resolution advisory.

    Returns:
        bool: True if pra and ra have the same parity, False otherwise.
    """
    return pra % 2 == ra % 2

def strengthen(pra, ra):
    """
    Check if ra (current resolution advisory) strengthens pra (previous resolution advisory).

    Args:
        pra (int): Previous resolution advisory.
        ra (int): Current resolution advisory.

    Returns:
        bool: True if ra is stronger than pra, False otherwise.
    """
    return ra > pra

def weaken(pra, ra):
    """
    Check if ra (current resolution advisory) weakens pra (previous resolution advisory).

    Args:
        pra (int): Previous resolution advisory.
        ra (int): Current resolution advisory.

    Returns:
        bool: True if ra is weaker than pra, False otherwise.
    """
    return ra < pra

def strongAlert(ra):
    """
    Determine if the resolution advisory (ra) represents a strong alert.

    Args:
        ra (int): Current resolution advisory.

    Returns:
        bool: True if ra represents a strong alert, False otherwise.
    """
    return ra > 2

# Reward function for HCAS_MDP
def reward(mdp, s, ra):
    """
    Compute the reward for a given state and resolution advisory in the HCAS_MDP.

    Args:
        mdp (object): HCAS_MDP instance containing currentTau.
        s (tuple): State represented as (r, t, p, vown, vint, pra):
            r (float): Distance between ownship and intruder.
            t (float): Angle to intruder relative to ownship heading.
            p (float): Heading angle of the intruder.
            vown (float): Velocity of the ownship.
            vint (float): Velocity of the intruder.
            pra (int): Previous resolution advisory.
        ra (int): Current resolution advisory.

    Returns:
        float: Reward value for the given state and resolution advisory.
    """
    r, t, p, vown, vint, pra = s
    tau = mdp.currentTau
    rew = 0.0

    # Compute relative positions and velocities
    relx = r * np.cos(t)
    rely = r * np.sin(t)
    dx = vint * np.cos(p) - vown
    dy = vint * np.sin(p)

    # Compute dv^2, tCPA (time to closest point of approach), and dCPA (distance at CPA)
    dv2 = dx**2 + dy**2
    dCPA = r
    tCPA = 0.0

    if dv2 > 0.0:
        tCPA = (-relx * dx - rely * dy) / dv2
        xT = relx + dx * tCPA
        yT = rely + dy * tCPA
        if tCPA > 0.0:
            dCPA = np.sqrt(xT**2 + yT**2)
        else:
            tCPA = 0.0

    # Penalty based on distance r when tau == 0
    if tau == 0.0:
        if r <= 500.0:
            rew -= 1.0
        else:
            rew -= 1.0 * np.exp(-(r - 500.0) / 500.0)

    # Scaling factor when pra is not COC
    factor = 0.1 if pra != COC else 1.0

    # Additional penalties when ra is not COC
    if ra != COC:
        rew -= 5e-4 * factor

        if strongAlert(ra):
            rew -= 2e-3 * factor

        if pra != COC and (not sameSense(pra, ra)):
            rew -= 5e-2  # High penalty for conflicting sense of direction
        elif strengthen(pra, ra):
            rew -= 1e-3  # Smaller penalty for strengthening advisory
        elif weaken(pra, ra):
            rew -= 2e-4  # Minor penalty for weakening advisory
    else:
        # Penalty based on dCPA and tCPA when ra == COC
        rew -= 1e-2 * np.exp(-dCPA / 500.0) * np.exp(-tCPA / 10.0) / factor

    return rew
