import numpy as np
from scipy.fftpack import fft, fftfreq

class PeriodicityDetector:
    def __init__(self, top_k_seasons=3):
        self.top_k_seasons = top_k_seasons


    def detect_periodicity(self, data):
        fft_series = fft(data)
        power = np.abs(fft_series)
        sample_freq = fftfreq(fft_series.size)

        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]

        # 取功率最大的前K个频率
        top_k_idxs = np.argpartition(powers, -self.top_k_seasons)[-self.top_k_seasons:]
        top_k_power = powers[top_k_idxs]
        fft_periods = (1 / freqs[top_k_idxs]).astype(int)

        print(f"top_k_power: {top_k_power}")
        print(f"fft_periods: {fft_periods}")

        # 判断周期性
        if any(top_k_power > np.mean(power) * 1.5):  # 假设显著的频率成分的功率至少是平均功率的1.5倍
            return "periodic"
        else:
            return "non-periodic"


# 测试代码
if __name__ == "__main__":
    # 创建周期性检测器实例
    detector = PeriodicityDetector(top_k_seasons=3)

    # 生成周期性时间序列数据
    time = np.linspace(0, 1, 500)
    periodic_data = np.sin(2 * np.pi * 5 * time)  # 5Hz 正弦波

    # 检测周期性
    result = detector.detect_periodicity(periodic_data)
    print(f"Periodic signal classified as: {result}")

    # 生成非周期性时间序列数据
    non_periodic_data = np.random.rand(500)

    # 检测周期性
    result = detector.detect_periodicity(non_periodic_data)
    print(f"Non-periodic signal classified as: {result}")
