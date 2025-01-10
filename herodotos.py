# Tworzenie historgramu z wygenerowanego pikla kanałów jonowych

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from scipy import stats

# parametry:
# pikel - tablica pikli | glob?
# her1 = Herodotos(pikel, x, y, z)
# her1.naive()

class Herodotos:
    """Herodotos
    ============
    A class for analyzing ion channel signals using various methods.


    Main features:
    1. Naive threshold detection between signal states
    2. Derivative-based state detection
    3. MDL (Minimum Description Length) analysis
    4. Deep learning approach for state identification
    5. Visualization tools for signal analysis
    """
    
    def __init__(self, data, bins=50):
        """Herodotos class constructor.

        
        Parameters
        ---------
        - data : `dict`
            Dictionary with measurement data in the form:
            - 'x': `np.array`
                Values of the ionic current
            - 'dwell times': `np.array`
        
        Attributes
        --------
        - data : `dict`
            Dictionary with measurement data
        - current : `np.array`
            Values of the ionic current
        - dwell_times : `np.array`
            Dwell times between states
        - breaks : `np.array`
            Cumulative sum of dwell times
        - T : `np.array`
            Time points
        - threshold : `np.float64`
            Value of the threshold between two peaks in the signal
        - bins : `int`
            Number of bins in the histogram
        - counts : `np.array`
            Number of counts in each bin
        - bin_edges : `np.array`
            Edges of the bins
        - bin_centers : `np.array`
            Centers of the bins
        - counts_smooth : `np.array`
            Smoothed counts with a Gaussian filter with sigma equal to 3
        """

        self.data = data
        self.current = np.array(data['x'])
        self.dwell_times = np.array(data['dwell times'])
        self.breaks = np.cumsum(self.dwell_times)
        self.T = np.arange(len(self.current))
        self.threshold = None
        self.bins = bins
        self.counts, self.bin_edges = np.histogram(self.current, bins=self.bins)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.counts_smooth = gaussian_filter1d(self.counts, sigma=3)

        
    def naive(self) -> np.float64:
        """Identifies the threshold value between two peaks in the signal.
        -----


        This method finds the two highest peaks in the `self.counts_smooth` array and 
        determines the threshold value between them by identifying the valley 
        closest to the midpoint of the two peaks.
        Raises:
            ValueError: If there are fewer than two peaks in the signal.
            ValueError: If no valleys are found between the two peaks.
        Attributes:
            self.threshold (np.float64): The bin center value at the identified threshold.
        Returns:
            threshold (np.float64): The identified threshold value.
        Examples:
        >>> herodotos = Herodotos(data)
        >>> result = herodotos.naive()
        """
        
        peaks, _ = signal.find_peaks(self.counts_smooth, distance=self.bins/5) # bardzo dziwaczne zachowanie. powiązane z bins?
        
        assert len(peaks) >= 2, "Sygnał nie ma dwóch maksimów"
        peak1, peak2 = np.sort(peaks)[-2:]

        inverted_counts = -self.counts_smooth
        valleys, _ = signal.find_peaks(inverted_counts, distance=1)
        possible_valleys = [v for v in valleys if peak1 < v < peak2]
        print(possible_valleys)

        if not possible_valleys:
            raise ValueError("Nie znaleziono dolin")
        else:
            middle = (peak1 + peak2) // 2
            difference = np.inf
            for valley in possible_valleys:
                new_difference = abs(middle - valley)
                if new_difference < difference:
                    difference = new_difference
                    self.threshold = self.bin_centers[valley]
                    
            print(f"Znaleziony threshold: {self.threshold}")
            return self.threshold




        
    
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

        TODO:Użyć SEABORN KDE
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
