import numpy as np

# ADVISORY INDICES
COC = 0
WL = 1
WR = 2
SL = 3
SR = 4

# State Type:
# In Python, we'll use a tuple structure for stateType, similar to Tuple{Float64, Float64, Float64, Float64, Float64, Int} in Julia
stateType = tuple  # This will represent a tuple of (Float64, Float64, Float64, Float64, Float64, Int) in Julia
actType = int
ACTIONS = [COC, WL, WR, SL, SR]

# Default parameters
discount_f = 1.0

# STATE CUTPOINTS
from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt

'''
# Con questo più situazioni critiche, maggioranza SL, SR

# Parametri della distribuzione
mu = 0            # Media
#std_theta, std_psi = 1.82, 0.86           # Deviazione standard
std_theta, std_psi = 1.9, 1          # Deviazione standard

sample_size = 41  # Numero di campioni
min_val = -np.pi  # Limite inferiore
max_val = np.pi   # Limite superiore

# Calcola i parametri della distribuzione troncata
a_t, b_t = (min_val - mu) / std_theta, (max_val - mu) / std_theta  # Limiti in termini di deviazioni standard

# Calcola i parametri della distribuzione troncata
a_p, b_p = (min_val - mu) / std_psi, (max_val - mu) / std_psi  # Limiti in termini di deviazioni standard


# Genera campioni dalla distribuzione normale troncata
samples_t = truncnorm.rvs(a_t, b_t, loc=mu, scale=std_theta, size=sample_size)
samples_p = truncnorm.rvs(a_p, b_p, loc=mu, scale=std_psi, size=sample_size)
'''
import numpy as np
from scipy.stats import truncnorm

#Con questo codice distibuzione circa pari per ogni azione e speculare

# Parametri della distribuzione
mu = 0  # Media
std_theta, std_psi = 1.2, 1.2  # Deviazione standard
sample_size = 41  # Numero di campioni (deve essere dispari per includere l'estremo centrale)
min_val = -np.pi  # Limite inferiore
max_val = np.pi   # Limite superiore

# Controllo sample_size sia sufficiente per includere gli estremi e il centro
if sample_size < 3:
    raise ValueError("Il sample_size deve essere almeno 3 per includere gli estremi e il centro.")

# Calcola i parametri della distribuzione troncata
a_t, b_t = (min_val - mu) / std_theta, (max_val - mu) / std_theta  # Limiti in termini di deviazioni standard
a_p, b_p = (min_val - mu) / std_psi, (max_val - mu) / std_psi  # Limiti in termini di deviazioni standard

# Numero di campioni casuali da generare (escludendo estremi e il valore centrale)
half_samples = (sample_size - 3) // 2  # 3 per gli estremi e il centro

# Genera campioni dalla distribuzione normale troncata
samples_t = truncnorm.rvs(a_t, b_t, loc=mu, scale=std_theta, size=half_samples)
samples_p = truncnorm.rvs(a_p, b_p, loc=mu, scale=std_psi, size=half_samples)

# Aggiungi gli estremi e il centro
samples_t = np.concatenate(([min_val], samples_t, [mu], -samples_t, [max_val]))
samples_p = np.concatenate(([min_val], samples_p, [mu], -samples_p, [max_val]))

# Risultati
print("Distribuzione specchiata theta:", samples_t)
print("Distribuzione specchiata psi:", samples_p)


# ranges aggiustati per avere più casi vicini
RANGES = list(np.geomspace(0.1, 56000.0, 32))
THETAS = np.sort(samples_t)  # Equivalent to LinRange in Julia
PSIS = np.sort(samples_p)

#RANGES = [0.0,25.0,50.0,75.0,100.0,150.0,200.0,300.0,400.0,500.0,510.0,750.0,1000.0,1500.0,2000.0,3000.0,4000.0,5000.0,7000.0,9000.0,11000.0,13000.0,15000.0,17000.0,19000.0,21000.0,25000.0,30000.0,35000.0,40000.0,48000.0,56000.0]
#THETAS = np.linspace(-np.pi, np.pi, 41)  # Equivalent to LinRange in Julia
#PSIS = np.linspace(-np.pi, np.pi, 41)

OWNSPEEDS = [200.0]  # [100.0, 200.0, 300.0, 400.0]  # ft/s
INTRSPEEDS = [200.0]  # [100.0, 200.0, 300.0, 400.0]  # ft/s

# You need to define LocalGIFunctionApproximator and RectangleGrid
# interp = LocalGIFunctionApproximator(RectangleGrid(RANGES, THETAS, PSIS, OWNSPEEDS, INTRSPEEDS, ACTIONS))

# Transition probabilities and turns
probs = [0.5, 0.25, 0.25]
turns = {
    COC: ([0.34, 0.33, 0.33], np.array([0.0, 1.5, -1.5]) * (np.pi / 180.0)),
    WL: (probs, np.array([1.5, 2.0, 1.25]) * (np.pi / 180.0)),
    WR: (probs, np.array([-1.5, -1.25, -2.0]) * (np.pi / 180.0)),
    SL: (probs, np.array([3.0, 4.0, 2.0]) * (np.pi / 180.0)),
    SR: (probs, np.array([-3.0, -2.0, -4.0]) * (np.pi / 180.0)),
    -1: ([0.34, 0.33, 0.33], np.array([0.0, 1.5, -1.5]) * (np.pi / 180.0))
}
