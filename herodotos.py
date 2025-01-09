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
        self.threshold = None
        self.counts, self.bin_edges = np.histogram(self.current, bins=50)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        
    def naive(self):
        """
        Metoda naiwna do detekcji stanów kanału jonowego.
        """
        peaks, _ = signal.find_peaks(self.counts, distance=5)
        print(peaks)
        if len(peaks) != 2:
            raise ValueError("Sygnał nie ma dwóch maksimów")
        else:
            two_highest_peaks = np.sort(peaks)[-2:]
            peak1, peak2 = np.sort(two_highest_peaks)

            inverted_counts = -self.counts
            valleys, _ = signal.find_peaks(inverted_counts, distance=2)

            possible_valleys = [v for v in valleys if peak1 < v < peak2]

            if not possible_valleys:
                raise ValueError("Nie znaleziono dolin")
            else:
                valley = min(possible_valleys, key=lambda x: self.counts[x])
                self.threshold = self.bin_centers[valley]
                print(f"Znaleziony threshold: {self.threshold}")


        
    
    def derevative(self)-> np.float64:
        """
        Metoda do detekcji stanów kanału jonowego na podstawie pochodnej sygnału.
        
        Zwraca
        ------
        result : np.float64
            Wartość prądu, która dzieli stany kanału jonowego na dwie grupy
        """
        derivative = np.gradient(self.current)
        mean_derivative = np.mean(derivative)

        return mean_derivative
    

    def mdl(self):
        pass

    def deep_learning(self):
        pass

    def plot(self):
        """
        Metoda do rysowania histogramu z kanałów jonowych.

        Parametry
        ---------
        result : np.float64
            Wynik analizy
        method : str
            Metoda analizy

        Zwraca
        ------
            None

        Przykład
        -------
            herodotos.plot(result, method='naive')
        """

        plt.figure(figsize=(8, 4))
        plt.bar(self.bin_edges[:-1], self.counts, width=np.diff(self.bin_edges), alpha=0.7, color='lightblue')
        if self.threshold is not None:
            plt.axvline(self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold:.2f}')
        plt.xlabel('Prąd')
        plt.ylabel('Liczba wystąpień')
        plt.title('Histogram z naniesionym progiem rozdziału')
        plt.legend()
        plt.show()






# Wczytanie danych
with open('data/simulation_m20_D0.001.p', 'rb') as f:
    data = pickle.load(f)

# Utworzenie obiektu klasy Herodotos
herodotos = Herodotos(data)

# # Analiza metodą naiwną
# result = herodotos.naive()
# herodotos.plot(result, method='naive')

herodotos.naive()


# Zamienić na:
# herodotos = Herodotos(data)
# 
# herodotos.naive()
# herodotos.plot()
# 
# herodotos.mdl()
# herodotos.plot()
# 
# herodotos.deep_learning()
# herodotos.plot()
