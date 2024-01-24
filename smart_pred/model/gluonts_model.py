import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.mx import SimpleFeedForwardEstimator, DeepStateEstimator, NBEATSEstimator, LSTNetEstimator, \
    WaveNetEstimator
from gluonts.torch import DeepAREstimator  # 可以添加更多的模型
from gluonts.mx.trainer import Trainer as GluonTrainer

from smart_pred.model.base import Basic_model


class GluonTS_model(Basic_model):
    def __init__(self, name="SimpleFeedForward", scaler=None, model_parameters=None):
        super().__init__(name, scaler, model_parameters)
        self.model = None
        self.name = name
        self.scaler = scaler
        self.default_model_parameters = {
            "pred_len": 24,
            "seq_len": 100,
            "freq": "1min",
        }
        self.model_parameters = model_parameters or self.default_model_parameters

        print(f"初始化 {self.name}!")

    def train(self, history, extra_parameters=None):
        history = np.array(history)
        start = pd.Period("2022-01-01 00:00", freq=self.model_parameters["freq"])

        # 创建 GluonTS dataset
        train_ds = ListDataset(
            [{"target": history[:-self.model_parameters["pred_len"]], "start": start}],
            freq=self.model_parameters["freq"]
        )

        # 选择模型并训练
        if self.name == "SimpleFeedForward":
            # trainer
            trainer = GluonTrainer(epochs=5, )
            self.model = SimpleFeedForwardEstimator(
                num_hidden_dimensions=[10],
                prediction_length=self.model_parameters["pred_len"],
                context_length=self.model_parameters["seq_len"],
                trainer=trainer
            )
        elif self.name == "DeepAR":
            self.model = DeepStateEstimator(
                freq=self.model_parameters["freq"],
                prediction_length=self.model_parameters["pred_len"],
                context_length=self.model_parameters["seq_len"],
                trainer_kwargs={"max_epochs": 5}
            )
        elif self.name == "NBEATS":
            trainer = GluonTrainer(epochs=5, )
            self.model = NBEATSEstimator(
                freq=self.model_parameters["freq"],
                prediction_length=self.model_parameters["pred_len"],
                context_length=self.model_parameters["seq_len"],
                trainer=trainer
            )
        elif self.name == "LSTNet":
            trainer = GluonTrainer(epochs=5, )
            self.model = LSTNetEstimator(
                prediction_length=self.model_parameters["pred_len"],
                context_length=self.model_parameters["seq_len"],
                trainer=trainer,
                ar_window=24,
                skip_size=24,
                channels=100,
                num_series=1,
                kernel_size=6,
            )

        self.predictor = self.model.train(train_ds)

    def predict(self, history, predict_window, extra_parameters=None):
        # 历史数据长度至少是seq_len + pred_len
        if len(history) < self.model_parameters["seq_len"] + self.model_parameters["pred_len"]:
            raise ValueError(
                f"历史数据长度至少是 {self.model_parameters['seq_len'] + self.model_parameters['pred_len']}")

        # 确保predict_window与模型参数中的pred_len相同
        if predict_window != self.model_parameters["pred_len"]:
            raise ValueError(f"predict_window必须等于 {self.model_parameters['pred_len']}")

        # 从history中获取最后seq_len个数据
        history = np.array(history)
        history = history[-self.model_parameters["seq_len"]:]

        # 创建测试数据集
        start = pd.Period("2022-01-01 00:00", freq=self.model_parameters["freq"])
        test_ds = ListDataset(
            [{"target": history, "start": start}],
            freq=self.model_parameters["freq"]
        )

        # 使用模型进行预测
        predictions = self.predictor.predict(test_ds)

        # 获取预测结果
        forecast = next(iter(predictions))
        forecast_values = forecast.mean_ts

        pred = forecast_values[-self.model_parameters["pred_len"]:].to_numpy()

        return pred


def Test():
    # 用正弦函数生成数据，1000个点，周期为100
    x = np.linspace(0, 100, 1000)
    y = np.sin(x)

    history = y[:-100]
    real = y[-100:]

    # extra_parameters
    extra_parameters = {
        "seq_len": 200,
        "pred_len": 100,
        "freq": "1min",
    }
    # 分别测试不同的模型
    model_names = [
        # "SimpleFeedForward",
        # "DeepAR",
        # "NBEATS",
        "LSTNet",
    ]

    i = 0
    for model_name in model_names:
        print(f"第 {i} 次测试")

        print(f"测试 {model_name} 模型")
        model = GluonTS_model(name=model_name, model_parameters=extra_parameters)

        print(f"模型 {model_name} 初始化完成")

        model.train(history=history, extra_parameters=extra_parameters)

        predict = model.predict(history=history, predict_window=100, extra_parameters=extra_parameters)

        print(f"predict: {predict}")
        print(f"real: {real}")

        mae = np.mean(np.abs(predict - real))

        print(f"mae: {mae}")
        print(f"测试 {model_name} 模型结束")

        # 画出预测结果
        import matplotlib.pyplot as plt
        plt.plot(real, color="black")
        plt.plot(predict, color="red")
        plt.legend(["True values", "Predict values"], loc="upper left", fontsize="xx-large")
        # 保存图片
        plt.savefig(f"model_{model_name}_mae_{mae}.pdf")
        i += 1


if __name__ == "__main__":
    Test()
