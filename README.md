# Herodotos – Ion Channel Histogram Generator

Herodotos is a Python tool designed to analyze ion channel data and generate histograms that help identify the threshold between open and closed states. This simple yet powerful utility is ideal for researchers in biophysics and physiology who wish to gain insight into the gating kinetics of ion channels.

> [!NOTE]
> This code is free to use for any and all cases. If you want to support this work feel free to join in and make a PR.

### The premise
>*...and gods said in unison as if shanting the most sacred hymn: "Let there be a way of showing where the hell is is the border of open and closed Ion Channels.", and the world listened.*

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Ion channels play a critical role in cellular signaling and function. Determining the threshold between their open and closed states is essential for understanding their behavior and kinetics. Herodotos processes experimental data (e.g., voltage or current traces) to produce histograms that highlight the distribution of channel states, enabling researchers to pinpoint these thresholds more effectively.


---

## Features

- **Histogram Generation:** Converts raw ion channel data into informative histograms.
- **Threshold Identification:** Helps distinguish between closed and open state distributions.
- **Simplicity & Flexibility:** Easy to use out-of-the-box with options to customize the analysis for specific datasets.
- **Open Source:** Free to use, modify, and extend under the GPL-3.0 license.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/University-of-Silesia-Ion-Channel/Herodotos.git
   cd Herodotos
   ```
2. **Set Up a Virtual Environment (Recommended):**
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Herodotos is driven by two main scripts: `herodotos.py` (the core processing module) and `main.py` (the entry point).

To generate a histogram from your ion channel dataset, run:

```bash
python main.py --input path/to/your/datafile.csv --output output_histogram.png
```

> [!NOTE]
> Replace `path/to/your/datafile.csv` with the path to your actual data file. For more detailed usage options, refer to the comments within `herodotos.py`.

## License

Herodotos is licensed under the `GPL-3.0` license. This means you are free to use, modify, and distribute the code, provided that any derivative work is also distributed under the same license.

## Acknowledgements

If you use Herodotos in your research or projects, please consider citing this tool and acknowledging its contribution. Your feedback and contributions are highly appreciated.
