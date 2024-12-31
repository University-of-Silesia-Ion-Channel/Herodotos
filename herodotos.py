# Tworzenie historgramu z wygenerowanego pikla kanałów jonowych

import numpy as np
from numba import njit
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans

# parametry:
# pikel - tablica pikli | glob?
# her1 = Herodotos(pikel, x, y, z)
# her1.naive()

class Herodotos:
    """
    Klasa do analizy danych w postaci histogramu z kanałów jonowych.
    Implementuje różne metody detekcji stanów kanału jonowego.
    
    Parametry
    ---------
    data : dict
        Słownik zawierający dane pomiarowe w postaci:
        - 'x': np.array
            Wartości prądu jonowego
        - 'dwell times': np.array
            Czasy przebywania w poszczególnych stanach
            
    Atrybuty
    --------
    current : np.array
        Wartości prądu jonowego
    dwell_times : np.array
        Czasy przebywania w poszczególnych stanach
    breaks : np.array
        Skumulowane czasy przebywania (punkty przejścia między stanami)
    T : np.array
        Wektor czasu dla pomiarów
    """
    
    def __init__(self, data):
        """
        Inicjalizacja obiektu klasy Herodotos.
        
        Parametry
        ---------
        data : dict
            Słownik z danymi pomiarowymi
        
        Atrybuty
        --------
        data : dict
            Słownik z danymi pomiarowymi
        current : np.array
            Wartości prądu jonowego
        dwell_times : np.array
            Czasy przebywania w poszczególnych stanach
        breaks : np.array
            Skumulowane czasy przebywania (punkty przejścia między stanami)
        T : np.array
            Wektor czasu dla pomiarów
        """
        self.data = data
        self.current = np.array(data['x'])
        self.dwell_times = np.array(data['dwell times'])
        self.breaks = np.cumsum(self.dwell_times)
        self.T = np.arange(len(self.current))
        
    def naive(self) -> np.float64:
        """
        Metoda naiwna do analizy histogramu i wyznaczenia punktu podziału stanów otwartego i zamkniętego kanału jonowego.

        Parametry
        ---------
        current : np.array
            Wartości prądu jonowego
        dwell_times : np.array
            Czasy przebywania w poszczególnych stanach
            
        Wynik
        ------
            threshold : np.float64
            
        """
        
        threshold = np.mean(self.current)
        # how to create the beans
        
        # threshold = bins[mean[len(mean)/2]][0]
        
        return threshold
    
    def mdl(self):
        pass

    def deep_learning(self):
        pass

    def plot(self, result=None, method='naive'):
        """
        Wizualizacja wyników analizy.
        
        Parametry
        ---------
        result : dict, optional
            Wynik jednej z metod analizy (naive, mdl lub deep_learning)
        method : str, optional
            Nazwa metody użytej do analizy ('naive', 'mdl' lub 'deep_learning')
        """
        if result is None:
            if method == 'naive':
                result = self.naive()
            elif method == 'mdl':
                result = self.mdl()
            else:
                result = self.deep_learning()
        
        plt.figure(figsize=(12, 8))
        
        # Wykres oryginalnego sygnału
        plt.subplot(211)
        plt.plot(self.T, self.current, 'k-', alpha=0.5, label='Sygnał oryginalny')
        
        # Wykres wykrytych stanów
        for state in np.unique(result['assignments']):
            mask = result['assignments'] == state
            plt.plot(self.T[mask], self.current[mask], '.', label=f'Stan {state}')
        
        plt.xlabel('Czas')
        plt.ylabel('Prąd')
        plt.legend()
        plt.title(f'Analiza metodą: {method}')
        
        # Histogram wartości prądu
        plt.subplot(212)
        plt.hist(self.current, bins=50, alpha=0.5, color='gray')
        
        for state in result['states']:
            plt.axvline(state, color='r', linestyle='--')
        
        plt.xlabel('Prąd')
        plt.ylabel('Liczba wystąpień')
        plt.title('Histogram wartości prądu')
        
        plt.tight_layout()
        plt.show()


# Wczytanie danych
with open('data/simulation_p20_D100.0.p', 'rb') as f:
    data = pickle.load(f)

# Utworzenie obiektu klasy Herodotos
herodotos = Herodotos(data)

# Analiza metodą naiwną
result = herodotos.naive()
herodotos.plot(result, method='naive')

