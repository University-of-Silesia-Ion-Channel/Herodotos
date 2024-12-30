# Tworzenie historgramu z wygenerowanego pikla kanałów jonowych

import numpy as np
from numba import njit
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# parametry:
# pikel - tablica pikli | glob?
# her1 = Herodotos(pikel, x, y, z)
# her1.naive()
    
fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))  # 1 row, 2 columns

def herodotos(data, ax=None):
    a = pickle.load(open(data, 'rb'))
    x = a['x'][:50000]
    breaks = np.cumsum(a['dwell times'])
    print(breaks)
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    T = np.arange(len(x))

    # Generate toggling stepvalue
    toggle_value = np.round(x[0])
    stepvalue = np.full_like(x, toggle_value, dtype=np.int16)  # Start with -1
    for b in breaks:
        toggle_value *= -1  # Toggle between -1 and 1
        stepvalue[int(np.floor(b)):] = toggle_value  # Apply the new value from the breakpoint onward

    # Plot the main data
    ax.plot(T, x, label='Data', color='blue', linewidth=1)

    # Plot the stepvalue
    ax.plot(T, stepvalue, label='Step Value', color='orange', linestyle='--', linewidth=1)

    # Plot vertical lines for breakpoints
    for b in breaks:
        ax.axvline(x=b, color='k', linestyle='-', linewidth=1, label='Breakpoint' if b == breaks[0] else "")

    # Set titles and labels
    ax.set_title("MDL method", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('X', fontsize=12)

    # Create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    # Adjust layout
    # fig.tight_layout()

    # Return just the figure and axes (not np.array)
    return ax

herodotos('data/simulation_p20_D100.0.p', ax=ax1)
plt.show()
