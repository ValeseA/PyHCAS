import h5py
import pandas as pd
import numpy as np

# Percorsi ai file
file_julia_path = "TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00_julia.h5"
file_python_path = "TrainingData/HCAS_rect_TrainingData_v6_pra0_tau00.h5"

# Funzione per caricare i dati e le informazioni aggiuntive
def load_data(file_path):
    """Carica i dati da un file .h5"""
    with h5py.File(file_path, 'r') as file:
        x = file['X'][:]
        y = file['y'][:]
        means = file['means'][:]
        ranges = file['ranges'][:]
        min_inputs = file['min_inputs'][:]
        max_inputs = file['max_inputs'][:]
    return x, y, means, ranges, min_inputs, max_inputs

# Carica i dati dai due file
x_julia, y_julia, means_julia, ranges_julia, min_inputs_julia, max_inputs_julia = load_data(file_julia_path)
x_python, y_python, means_python, ranges_python, min_inputs_python, max_inputs_python = load_data(file_python_path)

# Determina i nomi delle colonne in base alla configurazione
useRect = True  # Cambia se necessario
oneSpeed = True  # Cambia se necessario

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

output_columns = [f"q_{i}" for i in range(y_julia.shape[1])]

# Crea i DataFrame per i dati
df_julia = pd.DataFrame(np.column_stack((x_julia, y_julia)), columns=input_columns + output_columns)
df_python = pd.DataFrame(np.column_stack((x_python, y_python)), columns=input_columns + output_columns)

# Scrivi i DataFrame in due fogli di lavoro (sheet) separati in un file Excel
with pd.ExcelWriter('output_data.xlsx', engine='openpyxl') as writer:
    df_julia.to_excel(writer, sheet_name='Data_Julia', index=False)
    df_python.to_excel(writer, sheet_name='Data_Python', index=False)

print("File Excel creato con successo come 'output_data.xlsx'")
