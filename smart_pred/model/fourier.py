# This file is used to define the fourier model

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
            "history_error_correct": False,
            "is_scaler": False,
            "use_future": True,
            "is_round": False,
        }
        # print(f"self.scaler:{self.scaler}")
        pass

    def predict(self, history, predict_window, extra_parameters=None):
        # 方法独有的参数
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

        defaultMinNumOfSpectrumItems = 3
        defaultMaxNumOfSpectrumItems = 100
        defaultHighFrequencyThreshold = 1 / (60.0 * 60.0)
        defaultLowAmplitudeThreshold = 1

        # defaultFFTMarginFraction = 0.1
        defaultFFTMarginFraction = 0
        defaultFFTMinValue = 0.01

        if minNumOfSpectrumItems == None or minNumOfSpectrumItems == 0:
            minNumOfSpectrumItems = defaultMinNumOfSpectrumItems
        if maxNumOfSpectrumItems == None or maxNumOfSpectrumItems == 0:
            maxNumOfSpectrumItems = defaultMaxNumOfSpectrumItems
        if highFrequencyThreshold == None or highFrequencyThreshold == 0:
            highFrequencyThreshold = defaultHighFrequencyThreshold
        if lowAmplitudeThreshold == None or lowAmplitudeThreshold == 0:
            lowAmplitudeThreshold = defaultLowAmplitudeThreshold
        if marginFraction == None or marginFraction == 0:
            marginFraction = defaultFFTMarginFraction

        x = history
        n_predict = predict_window

        # 开始逻辑
        X = fft.fft(x)

        sampleLength = len(X)
        sampleRate = 1.0 / 60
        frequencies = []
        amplitudes = []
        for k in range(len(X)):
            frequencie = float(k) * sampleRate / sampleLength  # 计算频域数据
            frequencies.append(frequencie)
            amplitude = abs(X[k]) / sampleLength
            amplitudes.append(amplitude)

            # 第一个是直流分量，去除
        # print(f"len(amplitudes):{len(amplitudes)}")
        amplitudes = amplitudes[1: int(len(amplitudes) / 2)]
        amplitudes.sort(reverse=True)  # 按振幅降序
        # print(f"振幅降序:{amplitudes}")

        if len(amplitudes) >= maxNumOfSpectrumItems:
            minAmplitude = amplitudes[maxNumOfSpectrumItems - 1]
        else:
            minAmplitude = amplitudes[len(amplitudes) - 1]

        if minAmplitude < lowAmplitudeThreshold:
            minAmplitude = lowAmplitudeThreshold
        if len(amplitudes) >= minNumOfSpectrumItems and amplitudes[minNumOfSpectrumItems - 1] < minAmplitude:
            minAmplitude = amplitudes[minNumOfSpectrumItems - 1]

        for k in range(len(X)):
            amplitude = abs(X[k]) / sampleLength
            if amplitude < minAmplitude and frequencies[k] > highFrequencyThreshold:  # 去除噪声
                X[k] = 0

                # print(f"在逆变换之前的X:{X}")
        # print(f"在逆变换之前的X[12]:{X[12]}")
        # print(f"在逆变换之前的X[24]:{X[24]}")
        # print(f"在逆变换之前的X[100]:{X[100]}")

        x = fft.ifft(X)  # 傅里叶逆变换

        # for i in x:
        #     print(f"{i} \n")
        # print(f"逆变换之后x[0]:{x[0]}")
        # print(f"逆变换之后x[12]:{x[12]}")
        # print(f"逆变换之后x[24]:{x[24]}")
        # print(f"逆变换之后x[100]:{x[100]}")

        nSamples = len(x)
        nSamplesPerCycle = int(n_predict * 60 * sampleRate)
        # print(f"nSamples:{nSamples}")
        # print(f"nSamplesPerCycle:{nSamplesPerCycle}")

        output_list = []
        for i in range(nSamples - nSamplesPerCycle, nSamples):
            a = x[i].real
            # print(a)
            if a <= 0.0:
                a = defaultFFTMinValue  # 如果小于等于0，就取 默认的 0.01
            output_list.append(a * (1.0 + marginFraction))
        # print(f"还原的序列为:{output_list}")
        # print(f"还原的序列len(x):{len(output_list)}")

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
            "history_error_correct": False,
            "is_scaler": False,
            "use_future": True,
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
        # print(f"p:{p}")
        x_notrend = x - p[0] * t  # detrended x # 非趋势项
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        # print(f"x_freqdom:{x_freqdom}")
        # print(f"len(x_freqdom):{len(x_freqdom)}")
        f = fft.fftfreq(n)  # frequencies
        # for item in f:
        #     print(f"frequencies:{item} \n")
        # fig = plt.figure()
        # plt.plot(f,"r")
        # plt.show()
        # plt.savefig("test.png")
        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(f[i]))
        # print(f"indexes:{indexes}")

        t = np.arange(0, n + n_predict)
        # print(f"n:{n}")
        # print(f"n_predict:{n_predict}")
        # print(f"len(t):{len(t)}")
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
        "history_error_correct": False,
    }
    result = my_model.rolling_predict(history=history, predict_window=10, use_future=False, extra_parameters=extra_parameters)
    print(f"result:{result}")
    print(f"len(result):{len(result)}")