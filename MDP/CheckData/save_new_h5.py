import h5py
import numpy as np

# Percorso al file HDF5 originale
tableFile = './Qtables/HCAS_oneSpeed_v6_diff_dists.h5'

# Parametri di configurazione
useRect = False  # Cambia se necessario
oneSpeed = True  # Cambia se necessario

# Definisce i nomi delle colonne in base alla configurazione
if useRect:
    if oneSpeed:
        input_columns = ['x', 'y', 'psi']
    else:
        input_columns = ['x', 'y', 'psi', 'v_own', 'v_int']
else:
    if oneSpeed:
        input_columns = ['rho', 'theta', 'psi']
    else:
        input_columns = ['rho', 'theta', 'psi', 'v_own', 'v_int']

# Legge i dati dal file HDF5 originale
with h5py.File(tableFile, 'r') as f:
    # Carica la Q-table
    Q = np.array(f['q']).T
    ranges = np.array(f['ranges'])
    thetas = np.array(f['thetas'])
    psis = np.array(f['psi'])

# Genera le combinazioni di input
#ranges = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 510.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 7000.0, 9000.0, 11000.0, 13000.0, 15000.0, 17000.0, 19000.0, 21000.0, 25000.0, 30000.0, 35000.0, 40000.0, 48000.0, 56000.0])
#thetas = np.linspace(-np.pi, np.pi, 41)
#psis = np.linspace(-np.pi, np.pi, 41)
vowns = [200.0]
vints = [200.0]

if useRect:
    if oneSpeed:
        X = np.array([[r * np.cos(t), r * np.sin(t), p] for p in psis for t in thetas for r in ranges])
    else:
        X = np.array([[r * np.cos(t), r * np.sin(t), p, vo, vi] for vi in vints for vo in vowns for p in psis for t in thetas for r in ranges])
else:
    if oneSpeed:
        X = np.array([[r, t, p] for p in psis for t in thetas for r in ranges])
    else:
        X = np.array([[r, t, p, vo, vi] for vi in vints for vo in vowns for p in psis for t in thetas for r in ranges])

# Calcola le dimensioni per dividere la Q-table in sottotabelle
acts = [0, 1, 2, 3, 4]
ns2 = len(ranges) * len(thetas) * len(psis)
ns3 = len(acts)
data_rows = ns2 * ns3
assert Q.shape[0] % (data_rows * len(vowns) * len(vints)) == 0, "La dimensione della tabella Q non corrisponde al numero di righe attese"

tau_count = Q.shape[0] // (data_rows * len(vowns) * len(vints))
full_data = []

# Aggiungi le colonne tau e pra agli input e combina con Q
for tau in range(tau_count):
    for pra in acts:
        tau_column = np.full((ns2, 1), tau)
        pra_column = np.full((ns2, 1), pra)
        X_with_meta = np.hstack((tau_column, pra_column, X))
        Q_slice = Q[(tau * data_rows + pra * ns2):(tau * data_rows + (pra + 1) * ns2)]
        data_with_Q = np.hstack((X_with_meta, Q_slice))
        full_data.append(data_with_Q)

full_data = np.vstack(full_data)

# Genera i nomi di tutte le colonne
all_columns = ['tau', 'pra'] + input_columns + [f"q_{i}" for i in range(Q.shape[1])]

# Crea un file HDF5 unico per salvare i dati
with h5py.File('full_output_data.h5', 'w') as output_file:
    output_file.create_dataset('data', data=full_data, compression="gzip", compression_opts=9)
    output_file.attrs['columns'] = np.string_(all_columns)

print("File HDF5 creato con successo come 'full_output_data.h5'")