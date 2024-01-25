from smart_pred.model.base import Basic_model
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import fft

# 过滤掉 warnings
import warnings

warnings.filterwarnings('ignore')


class Crane_dsp_model(Basic_model):
    def __init__(self, name="Crane_dsp_model", scaler=MinMaxScaler()):
        print(f"初始化 Crane_dsp_model!")
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "moving_window": 20,
            "is_scaler": False,
            "is_round": False,
        }
        pass

    def predict(self, history, predict_window, extra_parameters=None):
        # 首先尝试从额外参数中获取一些特定的参数值，如果没有提供，则设置为None
        try:
            minNumOfSpectrumItems = extra_parameters["minNumOfSpectrumItems"]
            maxNumOfSpectrumItems = extra_parameters["maxNumOfSpectrumItems"]
            highFrequencyThreshold = extra_parameters["highFrequencyThreshold"]
            lowAmplitudeThreshold = extra_parameters["lowAmplitudeThreshold"]
            marginFraction = extra_parameters["marginFraction"]
            defaultFFTMinValue = extra_parameters["defaultFFTMinValue"]
        except:
            minNumOfSpectrumItems = None
            maxNumOfSpectrumItems = None
            highFrequencyThreshold = None
            lowAmplitudeThreshold = None
            marginFraction = None
            defaultFFTMinValue = None

        # 设置参数的默认值
        defaultMinNumOfSpectrumItems = 3
        defaultMaxNumOfSpectrumItems = 100
        defaultHighFrequencyThreshold = 1 / (60.0 * 60.0)
        defaultLowAmplitudeThreshold = 1
        defaultFFTMarginFraction = 0
        defaultFFTMinValue = 0.01

        # 如果参数值未提供或为0，则使用默认值
        if minNumOfSpectrumItems is None or minNumOfSpectrumItems == 0:
            minNumOfSpectrumItems = defaultMinNumOfSpectrumItems
        if maxNumOfSpectrumItems is None or maxNumOfSpectrumItems == 0:
            maxNumOfSpectrumItems = defaultMaxNumOfSpectrumItems
        if highFrequencyThreshold is None or highFrequencyThreshold == 0:
            highFrequencyThreshold = defaultHighFrequencyThreshold
        if lowAmplitudeThreshold is None or lowAmplitudeThreshold == 0:
            lowAmplitudeThreshold = defaultLowAmplitudeThreshold
        if marginFraction is None or marginFraction == 0:
            marginFraction = defaultFFTMarginFraction

        x = history
        n_predict = predict_window

        # 对历史数据应用快速傅里叶变换(FFT)
        X = fft.fft(x)

        sampleLength = len(X)
        sampleRate = 1.0 / 60
        frequencies = []
        amplitudes = []

        # 遍历频域数据，计算频率和振幅
        for k in range(len(X)):
            frequencie = float(k) * sampleRate / sampleLength
            frequencies.append(frequencie)
            amplitude = abs(X[k]) / sampleLength
            amplitudes.append(amplitude)

        # 去除直流分量，并按振幅降序排序
        amplitudes = amplitudes[1: int(len(amplitudes) / 2)]
        amplitudes.sort(reverse=True)

        # 设置最小振幅阈值
        if len(amplitudes) >= maxNumOfSpectrumItems:
            minAmplitude = amplitudes[maxNumOfSpectrumItems - 1]
        else:
            minAmplitude = amplitudes[-1]

        if minAmplitude < lowAmplitudeThreshold:
            minAmplitude = lowAmplitudeThreshold
        if len(amplitudes) >= minNumOfSpectrumItems and amplitudes[minNumOfSpectrumItems - 1] < minAmplitude:
            minAmplitude = amplitudes[minNumOfSpectrumItems - 1]

        # 过滤噪声：去除低于最小振幅阈值和高于高频阈值的频率分量
        for k in range(len(X)):
            amplitude = abs(X[k]) / sampleLength
            if amplitude < minAmplitude and frequencies[k] > highFrequencyThreshold:
                X[k] = 0

        # 应用傅里叶逆变换
        x = fft.ifft(X)

        nSamples = len(x)
        nSamplesPerCycle = int(n_predict * 60 * sampleRate)

        output_list = []

        # 生成预测结果
        for i in range(nSamples - nSamplesPerCycle, nSamples):
            a = x[i].real
            if a <= 0.0:
                a = defaultFFTMinValue  # 对小于等于0的值应用默认最小值
            output_list.append(a * (1.0 + marginFraction))

        return output_list


class Icebreaker_model(Basic_model):
    def __init__(self, name="Icebreaker_model", scaler=MinMaxScaler()):
        print(f"初始化Icebreaker_model!")
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 240,
            "pred_len": 1,
            "is_scaler": False,
            "is_round": True,
        }
        # print(f"self.scaler:{self.scaler}")
        pass

    def predict(self, history, predict_window, extra_parameters=None):

        # 方法独有的参数
        try:
            harmonics = extra_parameters["harmonics"]
        except:
            harmonics = 10

        x = history
        n_predict = predict_window

        x = np.array(x)
        n = x.size
        n_harm = harmonics  # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # find linear trend in x

        x_notrend = x - p[0] * t  # detrended x # 非趋势项
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain

        f = fft.fftfreq(n)  # frequencies

        indexes = list(range(n))

        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(f[i]))

        t = np.arange(0, n + n_predict)

        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n  # amplitude
            phase = np.angle(x_freqdom[i])  # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        res = restored_sig + p[0] * t

        predict = res[-predict_window:]
        return list(predict)



if __name__ == "__main__":
    my_model = Crane_dsp_model()
    # my_model = Icebreaker_model()

    history = [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1]
    extra_parameters = {
        "seq_len": 15,
        "pred_len": 1,
    }
    result = my_model.rolling_predict(history=history, predict_window=10, extra_parameters=extra_parameters)
    print(f"result:{result}")
    print(f"len(result):{len(result)}")