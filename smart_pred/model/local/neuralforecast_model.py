from smart_pred.model.local.base import Basic_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

from neuralforecast.models import RNN, GRU, LSTM, TCN, DeepAR, DilatedRNN, MLP, NHITS, NBEATS, NBEATSx
from neuralforecast.models import TFT, VanillaTransformer, Informer, Autoformer, PatchTST, FEDformer, TimesNet
from neuralforecast import NeuralForecast

import matplotlib.pyplot as plt
import numpy as np




class NeuralForecast_model(Basic_model):

    def __init__(self, name="NHITS", scaler=StandardScaler()):
        super().__init__(name, scaler)
        self.model = None
        self.name = name
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "is_scaler": False,
            "is_round": False,
        }
        print(f"初始化 {self.name}!")

    def train(self, history, extra_parameters=None):
        if extra_parameters is None:
            extra_parameters = self.default_extra_parameters

        seq_len = extra_parameters["seq_len"]
        pred_len = extra_parameters["pred_len"]

        # 数据预处理
        if self.scaler and extra_parameters["is_scaler"]:
            history = self.scaler.fit_transform(history.reshape(-1, 1)).reshape(-1)

        # 准备训练数据
        Y_df = pd.DataFrame({'y': history, 'ds': pd.date_range(start='2000-01-01', periods=len(history), freq='H')})
        Y_df['unique_id'] = 'time_series'

        # 获取模型
        if self.name == "NHITS":
            self.model = NHITS(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "NBEATS":
            self.model = NBEATS(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "NBEATSx":
            self.model = NBEATSx(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "TFT":
            self.model = TFT(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "VanillaTransformer":
            self.model = VanillaTransformer(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "Informer":
            self.model = Informer(input_size=seq_len, h=pred_len, max_steps=500)
        elif self.name == "Autoformer":
            self.model = Autoformer(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "PatchTST":
            self.model = PatchTST(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "FEDformer":
            self.model = FEDformer(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "TimesNet":
            self.model = TimesNet(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "RNN":
            self.model = RNN(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "GRU":
            self.model = GRU(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "LSTM":
            self.model = LSTM(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "TCN":
            self.model = TCN(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "DeepAR":
            self.model = DeepAR(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "DilatedRNN":
            self.model = DilatedRNN(input_size=seq_len, h=pred_len, max_steps=100)
        elif self.name == "MLP":
            self.model = MLP(input_size=seq_len, h=pred_len, max_steps=100)
        else:
            raise Exception("模型名称错误！")
        # 可以添加更多模型的条件分支

        # 训练模型
        self.nf = NeuralForecast(models=[self.model], freq='H')
        self.nf.fit(df=Y_df)

    def predict(self, history, predict_window, extra_parameters=None):
        if extra_parameters is None:
            extra_parameters = self.default_extra_parameters

        if self.scaler and extra_parameters["is_scaler"]:
            history = self.scaler.fit_transform(history.reshape(-1, 1)).reshape(-1)

        # 准备预测数据
        Y_df = pd.DataFrame({'y': history, 'ds': pd.date_range(start='2000-01-01', periods=len(history), freq='H')})
        Y_df['unique_id'] = 'time_series'

        # 预测
        Y_hat_df = self.nf.predict(df=Y_df).reset_index()

        # 处理预测结果
        predictions = Y_hat_df[self.name].values

        # 只保留预测窗口的结果
        predictions = predictions[-predict_window:]

        # 返回scaler
        if self.scaler and extra_parameters["is_scaler"]:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)

        return predictions


def Test():
    # 用正弦函数生成数据，1000个点，周期为100
    x = np.linspace(0, 100, 1000)
    y = np.sin(x)

    seq_len = 200
    pred_len = 100

    history = y[:-pred_len]
    real = y[-pred_len:]

    # extra_parameters
    extra_parameters = {
        "seq_len": seq_len,
        "pred_len": pred_len,
        "is_scaler": True,
        "is_round": False,
    }

    # 分别测试不同的模型
    model_names = [
        # "NHITS",
        # "NBEATS",
        # "NBEATSx",
        # "TFT",
        # "VanillaTransformer",
        "DeepAR",
        # "Informer",
        # "PatchTST",
        # "Autoformer", # mps不能用
        # "FEDformer", # mps不能用
        # "TimesNet", # mps不能用
        "RNN",
        "GRU",
        "LSTM",
        "TCN",
        "DilatedRNN",
        "MLP",
    ]  # 您可以在此处添加其他模型名称

    i = 0
    for model_name in model_names:
        print(f"第 {i} 次测试")

        print(f"测试 {model_name} 模型")
        model = NeuralForecast_model(name=model_name)

        print(f"模型 {model_name} 初始化完成")

        model.train(history=history, extra_parameters=extra_parameters)

        predict = model.predict(history=history, predict_window=pred_len, extra_parameters=extra_parameters)

        print(f"predict: {predict}")
        print(f"real: {real}")

        mae = float(np.mean(np.abs(predict - real)))

        print(f"mae: {mae}")
        print(f"测试 {model_name} 模型结束")

        # 画出预测结果
        plt.plot(real, color="black")
        plt.plot(predict, color="red")
        plt.legend(["True values", "Predict values"], loc="upper left", fontsize="xx-large")
        # 保存图片
        mae = round(mae, 2)

        # save name
        save_name = f"model_{model_name}_mae_{mae}_seq_len_{seq_len}_pred_len_{pred_len}.png"
        # 新建一个tmp
        import os
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")
        save_path = f"./tmp/{save_name}"
        plt.savefig(save_path)

        plt.show()
        plt.clf()

        i += 1


if __name__ == "__main__":
    Test()