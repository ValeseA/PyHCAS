import numpy as np

# State transition function
def transition(mdp, s, ra):
    """
    Compute the next states and their probabilities based on the current state and resolution advisory.

    Args:
        mdp (object): The Markov Decision Process instance, containing turns and dynamics data.
        s (tuple): Current state represented as (r, t, p, vown, vint, pra):
            r (float): Distance between ownship and intruder.
            t (float): Relative angle to intruder from ownship's heading.
            p (float): Heading angle of the intruder.
            vown (float): Velocity of the ownship.
            vint (float): Velocity of the intruder.
            pra (int): Previous resolution advisory.
        ra (int): Current resolution advisory.

    Returns:
        tuple: A tuple containing:
            - nextStates (array): Array of possible next states.
            - nextProbs (array): Array of probabilities for each next state.
    """
    r, t, p, vown, vint, pra = s

    # Initialize arrays for next states and probabilities
    nextStates = np.empty((9,), dtype=object)  # 9 possible combinations of ownship and intruder actions
    nextProbs = np.zeros(9)
    next_pra = ra  # The new resolution advisory
    ind = 0  # Index for populating arrays

    # Retrieve sigma-point probabilities and turns for both ownship and intruder
    ownProbs, ownTurns = mdp.turns[pra]
    intProbs, intTurns = mdp.turns[-1]

    # Compute next states and their probabilities
    for i in range(3):  # Iterate over ownship actions
        for j in range(3):  # Iterate over intruder actions
            # Compute next state dynamics
            next_r, next_t, next_p, next_vown, next_vint = dynamics(
                r, t, p, vown, vint, ownTurns[i], intTurns[j], pra
            )
            nextStates[ind] = (next_r, next_t, next_p, next_vown, next_vint, next_pra)
            nextProbs[ind] = ownProbs[i] * intProbs[j]  # Joint probability of the actions
            ind += 1

    return nextStates, nextProbs

# Angle normalization function
def normalize_angle(angle):
    """
    Normalize an angle to the range [-π, π].

    Args:
        angle (float): The angle in radians.

    Returns:
        float: The normalized angle in the range [-π, π].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Dynamic equations
def dynamics(r, t, p, vown, vint, ownTurn, intTurn, ra):
    """
    Compute the new state based on the current state and applied actions.

    Args:
        r (float): Distance between ownship and intruder.
        t (float): Relative angle to intruder from ownship's heading.
        p (float): Heading angle of the intruder.
        vown (float): Velocity of the ownship.
        vint (float): Velocity of the intruder.
        ownTurn (float): Turn angle applied by ownship.
        intTurn (float): Turn angle applied by the intruder.
        ra (int): Resolution advisory (not used directly in dynamics).

    Returns:
        tuple: The new state represented as (r_new, t_new, p_new, vown, vint):
            r_new (float): New distance between ownship and intruder.
            t_new (float): New relative angle to intruder.
            p_new (float): New heading angle of the intruder relative to ownship.
            vown (float): Velocity of the ownship (unchanged).
            vint (float): Velocity of the intruder (unchanged).
    """
    # Compute initial positions of ownship and intruder
    x_own = 0.0
    y_own = 0.0
    x_int = r * np.cos(t)
    y_int = r * np.sin(t)

    # Compute displacements due to velocities
    dx_own = vown
    dy_own = 0.0
    dx_int = vint * np.cos(p)
    dy_int = vint * np.sin(p)

    # Update positions after one time step
    x_own += dx_own
    y_own += dy_own
    x_int += dx_int
    y_int += dy_int

    # Compute relative positions in the new frame
    x_int_new = x_int - x_own
    y_int_new = y_int - y_own

    # Compute new headings
    heading_own = ownTurn
    heading_int = p + intTurn

    # Compute new distance and angles
    r_new = np.sqrt(x_int_new**2 + y_int_new**2)  # New distance
    t_new = normalize_angle(np.arctan2(y_int_new, x_int_new) - heading_own)  # New relative angle
    p_new = normalize_angle(heading_int - heading_own)  # New heading angle of the intruder

    return r_new, t_new, p_new, vown, vint
