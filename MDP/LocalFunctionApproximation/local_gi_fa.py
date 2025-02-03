import numpy as np
from typing import List, Union, Optional
from GridInterpolations.gridInterpolations import *

# Definizione della classe
class LocalGIFunctionApproximator:
    def __init__(self, grid, gvalues: Optional[np.ndarray] = None):
        self.grid = grid
        if gvalues is None:
            gvalues = np.zeros(len(grid))
        self.gvalues = gvalues

    # Metodo per ottenere il numero di punti di interpolazione
    def n_interpolating_points(self) -> int:
        return len(self.grid)

    # Altri metodi come definito nel tuo codice
    def get_all_interpolating_points(self) -> np.ndarray:
        return vertices(self.grid)

    def get_all_interpolating_values(self) -> np.ndarray:
        return self.gvalues

    def get_interpolating_nbrs_idxs_wts(self, v: np.ndarray):
        return interpolants(self.grid, v)

    def compute_value(self, v: np.ndarray) -> float:
        return interpolate(self.grid, self.gvalues, v)

    def compute_value_list(self, v_list: List[np.ndarray]) -> List[float]:
        assert len(v_list) > 0, "The input list must not be empty"
        return [self.compute_value(pt) for pt in v_list]

    def set_all_interpolating_values(self, gvalues: np.ndarray):
        self.gvalues = np.copy(gvalues)

# Funzione di estensione dell'orizzonte finito
def finite_horizon_extension(lfa: LocalGIFunctionApproximator, hor: np.ndarray) -> LocalGIFunctionApproximator:
    cut_points = lfa.grid.cutPoints
    extended_grid = RectangleGrid(*cut_points, hor)
    return LocalGIFunctionApproximator(extended_grid)
