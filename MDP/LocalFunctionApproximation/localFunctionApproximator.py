import numpy as np
from typing import List, Tuple, Union, Callable
from abc import ABC, abstractmethod

# Definizione della classe base astratta
class LocalFunctionApproximator(ABC):
    """
    Classe base per gli approssimatori di funzioni locali. 
    Definisce le interfacce per le operazioni di interpolazione e approssimazione.
    """
    @abstractmethod
    def n_interpolating_points(self) -> int:
        """
        Restituisce il numero di punti di interpolazione utilizzati dall'approssimatore.
        """
        pass

    @abstractmethod
    def get_all_interpolating_points(self) -> np.ndarray:
        """
        Restituisce il vettore di punti che vengono utilizzati per l'interpolazione.
        """
        pass

    @abstractmethod
    def get_all_interpolating_values(self) -> np.ndarray:
        """
        Restituisce il vettore di tutti i valori di interpolazione (nello stesso ordine dei punti).
        """
        pass

    @abstractmethod
    def get_interpolating_nbrs_idxs_wts(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restituisce una tupla (indici, pesi) per gli interpolanti per un punto di query specifico `v`.
        """
        pass

    @abstractmethod
    def compute_value(self, v: np.ndarray) -> float:
        """
        Restituisce il valore della funzione in un punto di query `v`, basato sull'approssimatore locale.
        """
        pass

    @abstractmethod
    def set_all_interpolating_values(self, vals: np.ndarray):
        """
        Imposta i valori di tutti i punti di interpolazione al vettore di input `vals`.
        """
        pass

    @abstractmethod
    def finite_horizon_extension(self, hor: int):
        """
        Estende l'approssimatore lungo una nuova dimensione per consentire approssimazioni con orizzonte finito.
        """
        pass


# Funzioni di supporto (implementazione astratta o come segnaposto)
def n_interpolating_points(lfa: LocalFunctionApproximator) -> int:
    """
    Restituisce il numero di punti di interpolazione che l'approssimatore sta utilizzando.
    """
    return lfa.n_interpolating_points()


def get_all_interpolating_points(lfa: LocalFunctionApproximator) -> np.ndarray:
    """
    Restituisce i punti di interpolazione utilizzati dall'approssimatore.
    """
    return lfa.get_all_interpolating_points()


def get_all_interpolating_values(lfa: LocalFunctionApproximator) -> np.ndarray:
    """
    Restituisce i valori di interpolazione utilizzati dall'approssimatore.
    """
    return lfa.get_all_interpolating_values()


def get_interpolating_nbrs_idxs_wts(lfa: LocalFunctionApproximator, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restituisce una tupla di indici e pesi per gli interpolanti per un punto di query `v`.
    """
    return lfa.get_interpolating_nbrs_idxs_wts(v)


def compute_value(lfa: LocalFunctionApproximator, v: np.ndarray) -> float:
    """
    Restituisce il valore della funzione in un punto di query `v`.
    """
    return lfa.compute_value(v)


def set_all_interpolating_values(lfa: LocalFunctionApproximator, vals: np.ndarray):
    """
    Imposta tutti i valori di interpolazione a `vals`.
    """
    lfa.set_all_interpolating_values(vals)


def finite_horizon_extension(lfa: LocalFunctionApproximator, hor: int):
    """
    Estende l'approssimatore lungo una nuova dimensione per permettere approssimazioni con orizzonte finito.
    """
    return lfa.finite_horizon_extension(hor)


# Includi moduli specifici per approssimatori basati su griglia e k-NN
# (Supponendo che questi moduli siano già implementati in Python separatamente)
from .local_gi_fa import LocalGIFunctionApproximator
from .local_nn_fa import LocalNNFunctionApproximator
