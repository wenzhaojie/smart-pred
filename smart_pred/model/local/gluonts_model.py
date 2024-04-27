import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.mx import SimpleFeedForwardEstimator, NBEATSEstimator

from gluonts.torch import DeepAREstimator, DLinearEstimator, WaveNetEstimator, PatchTSTEstimator, LagTSTEstimator  # 可以添加更多的模型
from gluonts.mx.trainer import Trainer as GluonTrainer

from smart_pred.model.local.base import Basic_model


class GluonTS_model(Basic_model):
    def __init__(self, name="SimpleFeedForward", scaler=None):
        super().__init__(name, scaler)
        self.model = None
        self.name = name
        self.scaler = scaler
        self.default_model_parameters = {
            "pred_len": 24,
            "seq_len": 100,
            "freq": "1min",
        }

        print(f"初始化 {self.name}!")

    def train(self, history, extra_parameters=None):
        # 如果标准化
        if self.scaler and extra_parameters["is_scaler"]:
            history = self.scaler.fit_transform(history.reshape(-1, 1)).reshape(-1)

        history = np.array(history)
        start = pd.Period("2022-01-01 00:00", freq=extra_parameters["freq"])

        # 创建 GluonTS dataset
        train_ds = ListDataset(
            [{"target": history[:-extra_parameters["pred_len"]], "start": start}],
            freq=extra_parameters["freq"]
        )

        # 选择模型并训练
        if self.name == "SimpleFeedForward":
            # trainer
            trainer = GluonTrainer(epochs=5, )
            self.model = SimpleFeedForwardEstimator(
                num_hidden_dimensions=[10],
                prediction_length=extra_parameters["pred_len"],
                context_length=extra_parameters["seq_len"],
                trainer=trainer
            )
        elif self.name == "DeepAR":
            self.model = DeepAREstimator(
                freq=extra_parameters["freq"],
                prediction_length=extra_parameters["pred_len"],
                context_length=extra_parameters["seq_len"],
                trainer_kwargs={"max_epochs": 5}
            )
        elif self.name == "NBEATS":
            trainer = GluonTrainer(epochs=5, )
            self.model = NBEATSEstimator(
                freq=extra_parameters["freq"],
                prediction_length=extra_parameters["pred_len"],
                context_length=extra_parameters["seq_len"],
                trainer=trainer
            )
        elif self.name == "WaveNet":
            self.model = WaveNetEstimator(
                prediction_length=extra_parameters["pred_len"],
                freq=extra_parameters["freq"],
            )
        elif self.name == "DLinear":
            self.model = DLinearEstimator(
                prediction_length=extra_parameters["pred_len"],
                context_length=extra_parameters["seq_len"],
                trainer_kwargs={"max_epochs": 5}
            )

        elif self.name == "PatchTST":
            self.model = PatchTSTEstimator(
                prediction_length=extra_parameters["pred_len"],
                context_length=extra_parameters["seq_len"],
                patch_len=10,
                trainer_kwargs={"max_epochs": 5}
            )
        elif self.name == "LagTST":
            self.model = LagTSTEstimator(
                prediction_length=extra_parameters["pred_len"],
                context_length=extra_parameters["seq_len"],
                freq=extra_parameters["freq"],
                trainer_kwargs={"max_epochs": 5}
            )

        self.predictor = self.model.train(train_ds)

    def predict(self, history, predict_window, extra_parameters=None):
        # 历史数据长度至少是seq_len + pred_len
        if len(history) < extra_parameters["seq_len"] + extra_parameters["pred_len"]:
            raise ValueError(
                f"历史数据长度至少是 {extra_parameters['seq_len'] + extra_parameters['pred_len']}")

        # 确保predict_window与模型参数中的pred_len相同
        if predict_window != extra_parameters["pred_len"]:
            raise ValueError(f"predict_window必须等于 {extra_parameters['pred_len']}")

        # 从history中获取最后seq_len个数据
        history = np.array(history)
        history = history[-extra_parameters["seq_len"]:]

        # 创建测试数据集
        start = pd.Period("2022-01-01 00:00", freq=extra_parameters["freq"])
        test_ds = ListDataset(
            [{"target": history, "start": start}],
            freq=extra_parameters["freq"]
        )

        # 使用模型进行预测
        predictions = self.predictor.predict(test_ds)

        # 获取预测结果
        forecast = next(iter(predictions))
        forecast_values = forecast.mean_ts

        pred = forecast_values[-extra_parameters["pred_len"]:].to_numpy()

        return pred


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
        "freq": "1min",
    }
    # 分别测试不同的模型
    model_names = [
        "SimpleFeedForward",
        "DeepAR",
        "NBEATS",
        "WaveNet",
        "DLinear",
        "PatchTST",
        "LagTST",
        "DeepNPTS",
    ]

    i = 0
    for model_name in model_names:
        print(f"第 {i} 次测试")

        print(f"测试 {model_name} 模型")
        model = GluonTS_model(name=model_name)

        print(f"模型 {model_name} 初始化完成")

        model.train(history=history, extra_parameters=extra_parameters)

        predict = model.predict(history=history, predict_window=pred_len, extra_parameters=extra_parameters)

        print(f"predict: {predict}")
        print(f"real: {real}")

        mae = float(np.mean(np.abs(predict - real)))

        print(f"mae: {mae}")
        print(f"测试 {model_name} 模型结束")

        # 画出预测结果
        #
        import matplotlib.pyplot as plt
        plt.plot(real, color="black")
        plt.plot(predict, color="red")
        plt.legend(["True values", "Predict values"], loc="upper left", fontsize="xx-large")
        # 保存图片
        # mae 保留两位小数
        mae = round(mae, 2)
        plt.savefig(f"model_{model_name}_mae_{mae}_seq_len_{seq_len}_pred_len_{pred_len}.png")
        plt.show()
        # 清理画布
        plt.clf()


        i += 1


if __name__ == "__main__":
    Test()
