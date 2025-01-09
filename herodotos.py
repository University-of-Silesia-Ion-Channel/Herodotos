# Tworzenie historgramu z wygenerowanego pikla kanałów jonowych

import numpy as np
from numba import njit
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
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
        self.counts_smooth = gaussian_filter1d(self.counts, sigma=3)

        
    def naive(self):
        """
        Metoda naiwna do detekcji stanów kanału jonowego.
        """
        peaks, _ = signal.find_peaks(self.counts, distance=5) # bardzo dziwaczne zachowanie. powiązane z bins?
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
        """

       
        fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Histogram", "Pochodna histogramu"],
        horizontal_spacing=0.1
        )

        fig.add_trace(
        go.Bar(
            x=self.bin_centers,
            y=self.counts,
            name="Histogram (surowy)",
            marker_color="rgba(0, 0, 200, 0.3)"
        ),
        row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.bin_centers,
                y=self.counts_smooth,
                mode="lines",
                name="Histogram wygładzony",
                line=dict(color="red")
            ),
            row=1, col=1
        )


        if self.threshold is not None:
            fig.add_shape(
                type="line",
                x0=self.threshold,
                x1=self.threshold,
                y0=0,
                y1=max(self.counts)*1.05,
                line=dict(color="green", dash="dash"),
                row=1, col=1
            )

            fig.add_annotation(
                x=self.threshold,
                y=max(self.counts)*1.08, # zwiększenie odległości od osi 0Y
                text=f"Threshold={self.threshold:.2f}",
                showarrow=False,
                font=dict(color="green"),
                row=1, col=1
            )

            fig.add_shape(
                type="line",
                x0=min(self.bin_centers),
                x1=max(self.bin_centers),
                y0=0,
                y1=0,
                line=dict(color="gray", dash="dot"),
                row=1, col=2
            )


        fig.update_layout(
            width=1400,
            height=700,
            title_text="Analiza histogramu kanałów jonowych",
            showlegend=True
        )


        fig.show()


        # below_mask = (self.current <= self.threshold)
        # above_mask = (self.current >= self.threshold)

        # plt.figure(figsize=(10, 4))
        # plt.plot(self.T[below_mask], self.current[below_mask], '.', color='blue', label='Stan 1 (poniżej threshold)')
        # plt.plot(self.T[above_mask], self.current[above_mask], '.', color='orange', label='Stan 2 (powyżej threshold)')

        # plt.axhline(self.threshold, color='red', linestyle='--', label=f'Threshold = {self.threshold:.2f}')
        # plt.xlabel('Czas')
        # plt.ylabel('Prąd')
        # plt.title('Kanał jonowy – dwa stany na jednym wykresie')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()







# Wczytanie danych
with open('data/simulation_m20_D0.001.p', 'rb') as f:
    data = pickle.load(f)

# Utworzenie obiektu klasy Herodotos
herodotos = Herodotos(data)

# # Analiza metodą naiwną
# result = herodotos.naive()
# herodotos.plot(result, method='naive')

herodotos.naive()
herodotos.plot()


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
