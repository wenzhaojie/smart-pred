import numpy as np
from scipy.fft import fft
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kurtosis


def classify_time_series(time_series):
    # 标准化时间序列
    time_series = (time_series - np.mean(time_series)) / np.std(time_series)

    # 进行ADF平稳性检测
    result = adfuller(time_series)
    print(f"ADF统计量: {result[0]}, p值: {result[1]}")
    p_value = result[1]

    # 计算FFT并找出主导频率
    fft_result = fft(time_series)
    fft_magnitude = np.abs(fft_result)
    dominant_frequency = np.argmax(fft_magnitude[1:len(fft_magnitude) // 2]) + 1  # 跳过零频率
    print(f"主导频率: {dominant_frequency}")

    # 计算峰度以检测突增
    kurtosis_result = kurtosis(time_series)
    print(f"峰度: {kurtosis_result}")

    if p_value > 0.05:
        print("时间序列非平稳。")
        if dominant_frequency > 1:
            print("由于主导频率，被分类为周期性。")
            return "周期"  # Periodic
        else:
            print("由于缺乏主导频率，被分类为平滑。")
            return "平滑"  # Smooth
    else:
        print("时间序列平稳。")
        if kurtosis_result > 3:
            print("由于高峰度，被分类为突增。")
            return "突增"  # Bursty
        else:
            print("由于低峰度，被分类为稀疏。")
            return "稀疏"  # Sparse


# Example time series data
time_series_example = np.random.normal(0, 1, 100)  # Replace with real time series data

# Classify the time series
classification_result = classify_time_series(time_series_example)
print(classification_result)
