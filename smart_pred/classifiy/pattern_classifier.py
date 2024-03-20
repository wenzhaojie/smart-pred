import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.stats import kurtosis

class PeriodicityDetector:
    def __init__(self, top_k_seasons=3):
        self.top_k_seasons = top_k_seasons  # 最多考虑的季节性个数

    def detect_periodicity(self, data):
        # 对数据进行快速傅里叶变换
        fft_series = fft(data)
        power = np.abs(fft_series)  # 计算功率谱
        sample_freq = fftfreq(fft_series.size)  # 计算对应的频率

        pos_mask = np.where(sample_freq > 0)  # 仅考虑正频率
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]

        # 找出功率最大的前K个频率
        top_k_idxs = np.argpartition(powers, -self.top_k_seasons)[-self.top_k_seasons:]
        top_k_power = powers[top_k_idxs]
        fft_periods = (1 / freqs[top_k_idxs]).astype(int)  # 计算对应的周期

        # 使用更复杂的阈值判断逻辑，例如考虑峰度
        power_kurtosis = kurtosis(powers)
        threshold = np.median(power) + np.std(power)  # 设置阈值
        print(f"主要功率: {top_k_power}")
        print(f"周期: {fft_periods}")
        print(f"功率峰度: {power_kurtosis}")
        print(f"阈值: {threshold}")

        # 判断是否具有周期性
        if np.any(top_k_power > threshold) and power_kurtosis > 1:
            return "周期性"
        else:
            return "非周期性"


# 测试代码
if __name__ == "__main__":
    # 创建周期性检测器实例
    detector = PeriodicityDetector(top_k_seasons=1)

    # 生成周期性时间序列数据
    time = np.linspace(0, 1, 500)
    periodic_data = np.sin(2 * np.pi * 5 * time)  # 100个点为一个周期

    # 检测周期性
    result = detector.detect_periodicity(periodic_data)
    print(f"周期信号被分类为: {result}")

    # 生成非周期性时间序列数据
    non_periodic_data = np.random.rand(500)

    # 检测周期性
    result = detector.detect_periodicity(non_periodic_data)
    print(f"非周期信号被分类为: {result}")
