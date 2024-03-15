import numpy as np
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kurtosis


def classify_time_series(time_series):
    # Normalize the time series
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)

    # Test for stationarity
    result = adfuller(time_series)

    print(result)

    p_value = result[1]

    # Compute FFT and find dominant frequency
    fft_result = fft(time_series)
    fft_magnitude = np.abs(fft_result)
    dominant_frequency = np.argmax(fft_magnitude[1:len(fft_magnitude) // 2]) + 1  # Skip the zero frequency

    # Compute Kurtosis for detecting bursts
    kurtosis_result = kurtosis(time_series)

    if p_value > 0.05:
        if dominant_frequency > 1:
            return "周期"  # Periodic
        else:
            return "平滑"  # Smooth
    else:
        if kurtosis_result > 3:
            return "突增"  # Bursty
        else:
            return "稀疏"  # Sparse


# Example time series data
time_series_example = np.random.normal(0, 1, 100)  # Replace with real time series data
classification_result = classify_time_series(time_series_example)
print(classification_result)
