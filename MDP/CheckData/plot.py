import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Carica i dati (usa il tuo file Excel salvato precedentemente)
df_julia = pd.read_excel('output_data.xlsx', sheet_name='Data_Julia')
df_python = pd.read_excel('output_data.xlsx', sheet_name='Data_Python')

df_julia = df_julia[df_julia['psi'] == -0.25]
df_python = df_python[df_python['psi'] == -0.25]

# Definiamo colori personalizzati
custom_colors = ['#AEC6CF', '#77DD77', '#FFB347', '#FF6961', '#CDB5E3']  # Blu, Verde, Giallo, Rosa, Lilla
custom_cmap = ListedColormap(custom_colors)


def plot_best_actions(df, title, use_3d=False):
    """Genera un grafico delle migliori azioni per un dataset."""
    # Trova l'azione migliore per ogni riga
    q_columns = [col for col in df.columns if col.startswith('q_')]

    df['best_action'] = df[q_columns].idxmax(axis=1).str.extract('(\d+)').astype(int)
     
    # Definizione delle azioni e della legenda
    action_labels = ['COC', 'WL', 'WR', 'SL', 'SR']  # Etichette delle azioni
    cmap = ListedColormap(['blue', 'orange', 'green', 'red', 'purple'])  # Colori personalizzati
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)  # Per mappare i valori 0-4
    
    # Crea il grafico
    fig = plt.figure(figsize=(10, 8))
    if use_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df['x'], df['y'], df['psi'], c=df['best_action'], cmap=custom_cmap,norm = norm, s=1, alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('psi')
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(df['x'], df['y'], c=df['best_action'], cmap=custom_cmap, norm=norm, s=1, alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Aggiungi legenda e titolo
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(0, 5), label='Azione migliore')
    cbar.ax.set_yticklabels(action_labels)  # Sostituisci i numeri con i nomi delle azioni
    
    ax.set_title(title)
    plt.show()

# Grafici per Julia e Python
plot_best_actions(df_julia, 'Migliore azione - Julia', use_3d=True)  # Modifica use_3d=True per 3D
plot_best_actions(df_python, 'Migliore azione - Python', use_3d=True)

print(df_julia[['x', 'y']].describe())
print(df_python[['x', 'y']].describe())


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_voronoi_actions(df, title):
    """Crea un grafico a regioni Voronoi per rappresentare le azioni migliori."""
    # Preparazione dei dati
    points = df[['x', 'y']].to_numpy()
    actions = df['best_action'].to_numpy()
    
    # Crea il diagramma di Voronoi
    vor = Voronoi(points)
    
    # Crea il grafico
    fig, ax = plt.subplots(figsize=(10, 8))
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black', line_width=0.5)

    # Colora le regioni basandoti sull'azione migliore
    for region_idx, point_idx in enumerate(vor.point_region):
        if -1 in vor.regions[point_idx]:  # Ignora regioni infinite
            continue
        region = vor.regions[point_idx]
        polygon = [vor.vertices[i] for i in region]
        action = actions[region_idx]
        ax.fill(*zip(*polygon), color=custom_colors[action], alpha=0.6)

    # Aggiungi annotazioni
    action_labels = {0: 'COC', 1: 'WL', 2: 'WR', 3: 'SL', 4: 'SR'}
    for action, label in action_labels.items():
        action_points = df[df['best_action'] == action][['x', 'y']].mean()
        ax.text(action_points['x'], action_points['y'], label, fontsize=12, color='black', ha='center', va='center', fontweight='bold')

    # Personalizza il grafico
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.show()

# Grafici per Julia e Python
plot_voronoi_actions(df_julia, 'Zone di controllo - Julia')
plot_voronoi_actions(df_python, 'Zone di controllo - Python')
