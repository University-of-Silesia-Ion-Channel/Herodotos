import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import tensorflow as tf
import csv
import matplotlib.pyplot as plt


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

        self.calculate_dwell_times = None
        self.fit_exponential = None
        self.compare_methods = None
        self.plot_dwell_time_histogram = None
        self.visualize_method_comparison = None

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

        input_shape = reconstructed_model.input_shape

        # Handle models expecting sequence input (e.g., LSTM, CNN)
        if len(input_shape) >= 3:  # Input shape: (None, time_steps, features)
            time_steps = input_shape[1]
            features = input_shape[2] if len(input_shape) > 2 else 1

            # If time_steps is None (variable length), treat entire signal as one sample
            if time_steps is None:
                val_set = val_set.reshape(1, -1, features)
            else:
                # Truncate or pad the signal to fit time_steps
                samples = len(val_set) // time_steps
                val_set = val_set[:samples * time_steps].reshape(samples, time_steps, features)
        else:
            # For models that don't expect sequential input (e.g., dense layers)
            val_set = val_set.reshape(-1, input_shape[1])

        predictions = reconstructed_model.predict(val_set)
        return predictions

    def calculate_dwell_times(self, signal, threshold=None):
        """
        Calculate dwell times for a binary signal based on threshold.

        Parameters
        ----------
        signal : np.array
            The signal to analyze
        threshold : float, optional
            Threshold to binarize the signal. If None, use self.threshold

        Returns
        -------
        tuple
            (open_dwell_times, closed_dwell_times) - Arrays of dwell times in samples
        """
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Threshold not specified and not previously calculated")
            threshold = self.threshold

        # Binarize the signal
        binary_signal = (signal > threshold).astype(int)

        # Find transitions
        transitions = np.diff(binary_signal)

        # Get indices of transitions
        rising_edges = np.where(transitions == 1)[0] + 1
        falling_edges = np.where(transitions == -1)[0] + 1

        # Ensure we start with a complete event
        if binary_signal[0] == 1:  # If signal starts open
            if len(rising_edges) == len(falling_edges):
                pass  # All good, we have complete events
            elif len(falling_edges) > len(rising_edges):
                rising_edges = np.insert(rising_edges, 0, 0)  # Prepend a zero
        else:  # If signal starts closed
            if len(rising_edges) == len(falling_edges) + 1:
                falling_edges = np.append(falling_edges, len(signal) - 1)  # Append the last index
            elif len(rising_edges) == len(falling_edges):
                pass  # All good, we have complete events

        # Calculate dwell times
        open_dwell_times = []
        closed_dwell_times = []

        # If we start with an open state
        if binary_signal[0] == 1:
            for i in range(min(len(falling_edges), len(rising_edges))):
                if i == 0:
                    open_dwell_times.append(falling_edges[i])
                else:
                    open_dwell_times.append(falling_edges[i] - rising_edges[i - 1])

                if i < len(rising_edges) - 1:
                    closed_dwell_times.append(rising_edges[i] - falling_edges[i])
        # If we start with a closed state
        else:
            for i in range(min(len(falling_edges), len(rising_edges))):
                open_dwell_times.append(falling_edges[i] - rising_edges[i])

                if i < len(rising_edges) - 1:
                    closed_dwell_times.append(rising_edges[i + 1] - falling_edges[i])

        return np.array(open_dwell_times), np.array(closed_dwell_times)

    def fit_exponential(self, dwell_times):
        """
        Fit an exponential distribution to dwell times and return tau

        Parameters
        ----------
        dwell_times : np.array
            Array of dwell times

        Returns
        -------
        float
            Tau - the time constant of the exponential distribution
        """
        if len(dwell_times) < 2:
            return np.nan

        # MLE estimate of lambda for exponential distribution
        # lambda = 1/mean, and tau = mean = 1/lambda
        tau = np.mean(dwell_times)
        return tau

    def compare_methods(self, ground_truth_threshold=None, true_open_tau=None, true_closed_tau=None):
        """
        Compare the three methods (threshold detection, linear regression, deep learning)
        using mean squared error on dwell times.

        Parameters
        ----------
        ground_truth_threshold : float, optional
            The true threshold to use as ground truth
        true_open_tau : float, optional
            The true open state dwell time (tau)
        true_closed_tau : float, optional
            The true closed state dwell time (tau)

        Returns
        -------
        dict
            Dictionary with results of all methods
        """
        results = {}

        # If ground truth is provided, use it
        if ground_truth_threshold is not None:
            # Calculate ground truth dwell times
            gt_open_dwell, gt_closed_dwell = self.calculate_dwell_times(self.current, ground_truth_threshold)

            # Calculate ground truth tau values if not provided
            if true_open_tau is None:
                true_open_tau = self.fit_exponential(gt_open_dwell)
            if true_closed_tau is None:
                true_closed_tau = self.fit_exponential(gt_closed_dwell)

            results['ground_truth'] = {
                'threshold': ground_truth_threshold,
                'open_dwell_times': gt_open_dwell,
                'closed_dwell_times': gt_closed_dwell,
                'open_tau': true_open_tau,
                'closed_tau': true_closed_tau
            }

        # Method 1: Threshold Detection
        try:
            self.threshold_detection()
            thresh_open_dwell, thresh_closed_dwell = self.calculate_dwell_times(self.current)
            thresh_open_tau = self.fit_exponential(thresh_open_dwell)
            thresh_closed_tau = self.fit_exponential(thresh_closed_dwell)

            results['threshold_detection'] = {
                'threshold': self.threshold,
                'open_dwell_times': thresh_open_dwell,
                'closed_dwell_times': thresh_closed_dwell,
                'open_tau': thresh_open_tau,
                'closed_tau': thresh_closed_tau
            }

            if true_open_tau is not None and true_closed_tau is not None:
                results['threshold_detection']['open_tau_mse'] = (thresh_open_tau - true_open_tau) ** 2
                results['threshold_detection']['closed_tau_mse'] = (thresh_closed_tau - true_closed_tau) ** 2
                results['threshold_detection']['combined_mse'] = ((thresh_open_tau - true_open_tau) ** 2 +
                                                                  (thresh_closed_tau - true_closed_tau) ** 2) / 2
        except Exception as e:
            results['threshold_detection'] = {'error': str(e)}

        # Method 2: Linear Regression
        try:
            threshold_lr = self.lin_reg_1()
            lr_open_dwell, lr_closed_dwell = self.calculate_dwell_times(self.current, threshold_lr)
            lr_open_tau = self.fit_exponential(lr_open_dwell)
            lr_closed_tau = self.fit_exponential(lr_closed_dwell)

            results['linear_regression'] = {
                'threshold': threshold_lr,
                'open_dwell_times': lr_open_dwell,
                'closed_dwell_times': lr_closed_dwell,
                'open_tau': lr_open_tau,
                'closed_tau': lr_closed_tau
            }

            if true_open_tau is not None and true_closed_tau is not None:
                results['linear_regression']['open_tau_mse'] = (lr_open_tau - true_open_tau) ** 2
                results['linear_regression']['closed_tau_mse'] = (lr_closed_tau - true_closed_tau) ** 2
                results['linear_regression']['combined_mse'] = ((lr_open_tau - true_open_tau) ** 2 +
                                                                (lr_closed_tau - true_closed_tau) ** 2) / 2
        except Exception as e:
            results['linear_regression'] = {'error': str(e)}

        # Method 3: Deep Learning (if model provided)
        try:

                # Assuming deep learning model gives binary predictions
                dl_predictions = self.deep_learning("./models/DeepChannel.keras")

                # Calculate dwell times based on deep learning prediction
                dl_open_dwell = []
                dl_closed_dwell = []
                current_state = dl_predictions[0]
                current_dwell = 1

                for i in range(1, len(dl_predictions)):
                    if dl_predictions[i] == current_state:
                        current_dwell += 1
                    else:
                        if current_state == 1:  # Open state
                            dl_open_dwell.append(current_dwell)
                        else:  # Closed state
                            dl_closed_dwell.append(current_dwell)
                        current_state = dl_predictions[i]
                        current_dwell = 1

                # Add the last dwell time
                if current_state == 1:
                    dl_open_dwell.append(current_dwell)
                else:
                    dl_closed_dwell.append(current_dwell)

                dl_open_dwell = np.array(dl_open_dwell)
                dl_closed_dwell = np.array(dl_closed_dwell)

                dl_open_tau = self.fit_exponential(dl_open_dwell)
                dl_closed_tau = self.fit_exponential(dl_closed_dwell)

                results['deep_learning'] = {
                    'open_dwell_times': dl_open_dwell,
                    'closed_dwell_times': dl_closed_dwell,
                    'open_tau': dl_open_tau,
                    'closed_tau': dl_closed_tau
                }

                if true_open_tau is not None and true_closed_tau is not None:
                    results['deep_learning']['open_tau_mse'] = (dl_open_tau - true_open_tau) ** 2
                    results['deep_learning']['closed_tau_mse'] = (dl_closed_tau - true_closed_tau) ** 2
                    results['deep_learning']['combined_mse'] = ((dl_open_tau - true_open_tau) ** 2 +
                                                                (dl_closed_tau - true_closed_tau) ** 2) / 2

        except Exception as e:
            results['deep_learning'] = {'error': str(e)}

        return results

    def plot_dwell_time_histogram(self, dwell_times, tau=None, bins=50, title="Dwell Time Histogram"):
        """
        Plot histogram of dwell times with fitted exponential if tau is provided

        Parameters
        ----------
        dwell_times : np.array
            Array of dwell times
        tau : float, optional
            Time constant of the exponential distribution
        bins : int
            Number of bins for histogram
        title : str
            Title of the plot
        """
        if len(dwell_times) == 0:
            print(f"No dwell times to plot for {title}")
            return

        plt.figure(figsize=(10, 6))

        # Plot histogram
        counts, bin_edges, _ = plt.hist(dwell_times, bins=bins, density=True, alpha=0.6,
                                        label='Dwell times')

        # Plot fitted exponential if tau is provided
        if tau is not None and not np.isnan(tau):
            x = np.linspace(0, max(dwell_times), 1000)
            # PDF of exponential distribution with lambda = 1/tau
            y = (1 / tau) * np.exp(-(1 / tau) * x)
            plt.plot(x, y, 'r-', linewidth=2,
                     label=f'Exponential fit (τ = {tau:.2f})')

        plt.title(title)
        plt.xlabel('Dwell Time')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_method_comparison(self, results):
        """
        Visualize comparison of methods with bar plots of MSE and tau values

        Parameters
        ----------
        results : dict
            Results dictionary from compare_methods
        """
        methods = [method for method in results.keys() if method != 'ground_truth' and 'error' not in results[method]]

        if not methods:
            print("No valid methods to compare")
            return

        # Create figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Tau values comparison
        open_taus = [results[method]['open_tau'] for method in methods]
        closed_taus = [results[method]['closed_tau'] for method in methods]

        x = np.arange(len(methods))
        width = 0.35

        axs[0].bar(x - width / 2, open_taus, width, label='Open State Tau')
        axs[0].bar(x + width / 2, closed_taus, width, label='Closed State Tau')

        if 'ground_truth' in results:
            axs[0].axhline(y=results['ground_truth']['open_tau'], color='r', linestyle='--',
                           label='True Open Tau')
            axs[0].axhline(y=results['ground_truth']['closed_tau'], color='g', linestyle='--',
                           label='True Closed Tau')

        axs[0].set_xlabel('Method')
        axs[0].set_ylabel('Tau Value')
        axs[0].set_title('Comparison of Tau Values')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(methods)
        axs[0].legend()

        # MSE comparison
        if 'ground_truth' in results:
            open_mse = [results[method].get('open_tau_mse', np.nan) for method in methods]
            closed_mse = [results[method].get('closed_tau_mse', np.nan) for method in methods]
            combined_mse = [results[method].get('combined_mse', np.nan) for method in methods]

            axs[1].bar(x - width, open_mse, width, label='Open Tau MSE')
            axs[1].bar(x, closed_mse, width, label='Closed Tau MSE')
            axs[1].bar(x + width, combined_mse, width, label='Combined MSE')

            axs[1].set_xlabel('Method')
            axs[1].set_ylabel('Mean Squared Error')
            axs[1].set_title('Error Comparison')
            axs[1].set_xticks(x)
            axs[1].set_xticklabels(methods)
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, 'No ground truth provided for MSE calculation',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axs[1].transAxes)

        plt.tight_layout()
        plt.show()

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