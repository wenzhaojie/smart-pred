import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.mx import SimpleFeedForwardEstimator, Trainer
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

    def train(self, history, extra_parameters=None, model_parameters=None):
        history = np.array(history)
        start = pd.Period("2022-01-01 00:00", freq=self.model_parameters["freq"])

        # 创建 GluonTS dataset
        train_ds = ListDataset(
            [{"target": history[:-self.model_parameters["pred_len"]], "start": start}],
            freq=self.model_parameters["freq"]
        )

        # trainer
        trainer = GluonTrainer(epochs=5)

        # 选择模型并训练
        if self.name == "SimpleFeedForward":
            self.model = SimpleFeedForwardEstimator(
                num_hidden_dimensions=[10],
                prediction_length=self.model_parameters["pred_len"],
                context_length=self.model_parameters["seq_len"],
                trainer=trainer
            )
        elif self.name == "DeepAR":
            self.model = DeepAREstimator(
                prediction_length=self.model_parameters["pred_len"],
                context_length=self.model_parameters["seq_len"],
                trainer=trainer
            )
        # ... 添加其他模型的条件

        self.predictor = self.model.train(train_ds)

        # 开始训练
        self.model = self.model.train(train_ds)


    def predict(self, history, predict_window, extra_parameters=None):
        # 历史数据长度至少是seq_len + pred_len
        if len(history) < self.model_parameters["seq_len"] + self.model_parameters["pred_len"]:
            raise ValueError(f"历史数据长度至少是 {self.model_parameters['seq_len'] + self.model_parameters['pred_len']}")

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
    # extra_parameters
    extra_parameters = {
        "seq_len": 128,
        "pred_len": 10,
        "freq": "1min",
    }
    # 分别测试不同的模型
    model_names = [
        "SimpleFeedForward",
        "DeepAR",
    ]
    for model_name in model_names:
        print(f"测试 {model_name} 模型")
        model = GluonTS_model(name=model_name, model_parameters=extra_parameters)
        model.train(history=y, extra_parameters=extra_parameters)
        predict = model.predict(history=y, predict_window=10, extra_parameters=extra_parameters)
        print(f"predict: {predict}")

        print(f"测试 {model_name} 模型结束")


if __name__ == "__main__":
    Test()