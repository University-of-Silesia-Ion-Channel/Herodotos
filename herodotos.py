import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import tensorflow as tf
import csv


class Herodotos:
    """Herodotos
    ============
    A class for analyzing ion channel signals using various methods.


    Main features:
    1. Naive threshold detection between signal states
    2. Derivative-based state detection
    3. Deep learning approach for state identification
    4. Visualization tools for signal analysis
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
        if type(data) is np.ndarray:
            data = {'x': data}
        self.data = data
        self.current = np.array(data['x'])

        self.T = np.arange(len(self.current))

        self.threshold = None
        self.bins = bins

        self.counts, self.bin_edges = np.histogram(self.current, bins=self.bins)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.counts_smooth = gaussian_filter1d(self.counts, sigma=3)

        # Log scale
        self.log_counts = np.log(np.where(self.counts <= 0, 1e-12, self.counts))
        if np.any(self.bin_centers <= 0):
            self.shift = np.abs(np.min(self.bin_centers)) + 1e-12
            self.bin_centers_shifted = self.bin_centers + self.shift
        else:
            self.shift = 0.0
            self.bin_centers_shifted = self.bin_centers

        self.log_bin_centers = np.log(self.bin_centers_shifted)

        self.x1, self.x2 = None, None  # First peak
        self.y1, self.y2 = None, None  # Second peak
        self.xv, self.yv = 0, 0  # Valley

        self.vert1, self.vert2 = None, None

    def __find_peak_and_valley(self):
        peaks, _ = signal.find_peaks(self.counts_smooth,
                                     distance=self.bins / 5)

        assert len(peaks) >= 2, "Signal does not contain two maximums"
        peak1, peak2 = np.sort(peaks)[-2:]

        inverted_counts = -self.counts_smooth
        valleys, _ = signal.find_peaks(inverted_counts, distance=1)
        possible_valleys = [v for v in valleys if peak1 < v < peak2]
        return peak1, peak2, possible_valleys

    def __probability_of_open(self, threshold) -> np.float64:
        """Calculates the probability of the channel being open.
        -----
        Returns:
            probability (np.float64): The probability of the channel being open.
        """
        if threshold is not None:
            return np.float64(np.sum(self.current > threshold) / len(self.current))
        else:
            raise ValueError("Threshold has not been found")

    def threshold_detection(self) -> np.float64:
        """Naively identifies the threshold value between two peaks in the signal.
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
        >>> result = herodotos.threshold_detection()
        """
        peak1, peak2, possible_valleys = self.__find_peak_and_valley()

        if not possible_valleys:
            raise ValueError("Valleys have not been found")
        else:
            middle = (peak1 + peak2) // 2
            difference = np.inf
            for valley in possible_valleys:
                new_difference = abs(middle - valley)
                if new_difference < difference:
                    difference = new_difference
                    self.threshold = np.float64(self.bin_centers[valley])

            print(f"[Threshold detection:] Found threshold: {self.threshold}")
            return self.__probability_of_open(self.threshold)

    def lin_reg(self) -> np.float64:
        """
        Metoda do detekcji stanów kanału jonowego na podstawie pochodnej sygnału.
        
        Returns:
        treshold (np.float64): `x` value of intersection of the derevatives 
        """

        peak1, peak2, possible_valleys = self.__find_peak_and_valley()

        # TODO: Regresją liniową znaleźć punkt przecięcia obu peaków.
        #       Zrobić Kernel Density Estimation Epanechnikova
        #       Fitowanie power curve
        #       POWER LAW
        print(peak1, peak2)

        differences = np.diff(self.counts)
        middle = (peak1 + peak2) // 2
        print(middle)
        # <- middle -> 
        # jeśli np.diff od tego x-a będzie większy od średniej jak dotąd plus jakiś epsilon,
        # to flagujesz to jako wartość drugiego punktu do zrobienia prostej
        epsilon = 100
        foo, bar = 0, 0
        for i in range(middle, peak1, -1):
            # print(i)
            if np.abs(differences[i] - differences[i - 1]) > epsilon:
                foo = i - 1
                break
        for i in range(middle, peak2 + 1):
            # print(np.abs(differences[i] - differences[i-1]))
            if np.abs(differences[i] - differences[i - 1]) > epsilon:
                bar = i
                break

        self.vert1, self.vert2 = foo, bar

        #   _   _
        #   o\_/o
        #     x <- szukamy tego!!!

        return np.float64(1.0)

    def lin_reg_1(self) -> np.float64:
        # Zakładamy, że metoda __find_peak_and_valley() zwraca indeksy pików histogramu
        peak1, peak2, possible_valleys = self.__find_peak_and_valley()
        print("Piki:", peak1, peak2)

        # Obliczamy „pochodną” w przestrzeni log-log
        differences = np.diff(self.log_counts)

        # Obliczamy indeks środkowy pomiędzy pikami
        middle = (peak1 + peak2) // 2
        print("Indeks środka:", middle)

        # Ustalamy wartość progową dla wykrycia nagłej zmiany w przestrzeni log-log
        epsilon = 1.0  # epsilon dobrany eksperymentalnie – wartość w przestrzeni logarytmicznej
        foo, bar = None, None

        # Szukamy punktu foo – przesuwając się od środka w stronę lewego piku (peak1)
        for i in range(middle, peak1, -1):
            if np.abs(differences[i] - differences[i - 1]) > epsilon:
                foo = i - 1
                break
        if foo is None:
            foo = peak1
        print("foo (lewy punkt):", foo)

        # Szukamy punktu bar – przesuwając się od środka w stronę prawego piku (peak2)
        for i in range(middle, peak2 + 1):
            if np.abs(differences[i] - differences[i - 1]) > epsilon:
                bar = i
                break
        if bar is None:
            bar = peak2
        print("bar (prawy punkt):", bar)

        # Opcjonalnie zapisujemy wykryte punkty
        self.vert1, self.vert2 = foo, bar

        # Dla lewego zbocza – wybieramy punkty: (log_bin_centers[foo], log_counts[foo]) oraz (log_bin_centers[peak1], log_counts[peak1])
        x_left = np.array([self.log_bin_centers[foo], self.log_bin_centers[peak1]])
        y_left = np.array([self.log_counts[foo], self.log_counts[peak1]])
        if x_left[1] - x_left[0] == 0:
            m1 = 0.0
        else:
            m1 = (y_left[1] - y_left[0]) / (x_left[1] - x_left[0])
        b1 = y_left[0] - m1 * x_left[0]
        print(f"Lewa prosta: y = {m1:.3f}*x + {b1:.3f}")

        # Dla prawego zbocza – wybieramy punkty: (log_bin_centers[bar], log_counts[bar]) oraz (log_bin_centers[peak2], log_counts[peak2])
        x_right = np.array([self.log_bin_centers[bar], self.log_bin_centers[peak2]])
        y_right = np.array([self.log_counts[bar], self.log_counts[peak2]])
        if x_right[1] - x_right[0] == 0:
            m2 = 0.0
        else:
            m2 = (y_right[1] - y_right[0]) / (x_right[1] - x_right[0])
        b2 = y_right[0] - m2 * x_right[0]
        print(f"Prawa prosta: y = {m2:.3f}*x + {b2:.3f}")

        # Wyznaczamy punkt przecięcia dwóch prostych w przestrzeni log-log:
        if m1 == m2:
            # Gdy linie są równoległe, wybieramy logarytmiczny środek jako przybliżenie
            x_intersect_log = self.log_bin_centers[middle]
        else:
            x_intersect_log = (b2 - b1) / (m1 - m2)
        print("Punkt przecięcia w log skali (x):", x_intersect_log)

        # Przekształcamy wynik z przestrzeni log-log do oryginalnej skali:
        threshold = np.exp(x_intersect_log) - self.shift
        print("Threshold (próg) w skali oryginalnej:", threshold)

        return np.float64(threshold)

    def deep_learning(self, model):
        reconstructed_model = tf.keras.models.load_model(model)
        scaler = MinMaxScaler(feature_range=(0, 1))
        val_set = scaler.fit_transform(self.current.reshape(-1, 1))
        val_set = val_set.reshape()
        return reconstructed_model.predict(self.current)






    def plot(self, w=1000, h=400):
        """
        Metoda do rysowania histogramu z kanałów jonowych.

        TODO:Użyć SEABORN KDE
        """

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Histogram w/ threshold detection", "Logarithmic histogram w/ linear regression"],
            horizontal_spacing=0.1
        )
        # Threshold detection
        fig.add_trace(
            go.Bar(
                x=self.bin_centers,
                y=self.counts/np.sum(self.counts * np.diff(self.bin_edges)),
                name="Histogram (raw)",
                marker_color="rgba(0, 0, 200, 0.3)"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.bin_centers,
                y=self.counts_smooth/np.sum(self.counts_smooth * np.diff(self.bin_edges)),
                mode="lines",
                name="Smoothed histogram",
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
                y1=max(self.counts/np.sum(self.counts * np.diff(self.bin_edges))) * 1.05,
                line=dict(color="green", dash="dash"),
                row=1, col=1
            )

            fig.add_annotation(
                x=self.threshold,
                y=max(self.counts/np.sum(self.counts * np.diff(self.bin_edges))) * 1.08,  # zwiększenie odległości od osi 0Y
                text=f"Threshold={self.threshold:.2f}",
                showarrow=False,
                font=dict(color="green"),
                row=1, col=1
            )

        # Linear regression
        fig.add_trace(
            go.Scatter(
                x=self.bin_centers,
                y=self.counts/np.sum(self.counts * np.diff(self.bin_edges)),
                name="Logarithmic histogram",
            ), row=1, col=2
        )

        fig.update_xaxes(title_text="log(I)", type="log", row=1, col=2)
        fig.update_yaxes(title_text="log(PDF)", type="log", row=1, col=2)

        fig.update_layout(
            width=w,
            height=h,
            title_text="Analiza histogramu kanałów jonowych",
            showlegend=True
        )

        fig.show()

    def save(self, path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['bin_centers', 'counts'])
            for i in range(len(self.bin_centers)):
                writer.writerow([self.bin_centers[i], self.counts[i]])