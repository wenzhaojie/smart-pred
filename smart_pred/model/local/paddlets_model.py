from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from paddlets import TSDataset
from paddlets.models.forecasting import RNNBlockRegressor, LSTNetRegressor, MLPRegressor, NBEATSModel, TCNRegressor, NHiTSModel, TransformerModel, InformerModel, TFTModel, SCINetModel, DeepARModel


import time
from smart_pred.model.local.base import Basic_model


class PaddleTS_model(Basic_model):
    def __init__(self, name="NHiTSModel", scaler=StandardScaler()):
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
        # 转换np数据格式
        history = np.array(history)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            history = self.scaler.fit_transform(history.reshape(-1, 1)).reshape(-1)

        # 先初始化 dataframe
        out_chunk_len = extra_parameters["pred_len"]
        history_df = pd.DataFrame(
            {
                'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
                'value': history,
            },
        )
        train_dataset = TSDataset.load_from_dataframe(
            history_df,  # Also can be path to the CSV file
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )

        # 初始化模型
        if extra_parameters is None:
            self.extra_parameters = self.default_extra_parameters

        seq_len = extra_parameters["seq_len"]
        pred_len = extra_parameters["pred_len"]

        model_parameters = {
            "in_chunk_len": seq_len,
            "out_chunk_len": pred_len
        }

        # 根据模型name 来指定模型对象
        if self.name == "RNNBlockRegressor":
            self.model = RNNBlockRegressor(
                **model_parameters
            )
        elif self.name == "LSTNetRegressor":
            self.model = LSTNetRegressor(
                **model_parameters
            )
        elif self.name == "MLPRegressor":
            self.model = MLPRegressor(
                **model_parameters
            )
        elif self.name == "NBEATSModel":
            self.model = NBEATSModel(
                **model_parameters
            )
        elif self.name == "TCNRegressor":
            self.model = TCNRegressor(
                **model_parameters
            )
        elif self.name == "NHiTSModel":
            self.model = NHiTSModel(
                **model_parameters
            )
        elif self.name == "TransformerModel":
            self.model = TransformerModel(
                **model_parameters
            )
        elif self.name == "InformerModel":
            self.model = InformerModel(
                **model_parameters
            )
        elif self.name == "DeepARModel":
            self.model = DeepARModel(
                **model_parameters
            )
        elif self.name == "TFTModel":
            self.model = TFTModel(
                **model_parameters
            )
        elif self.name == "SCINetModel":
            self.model = SCINetModel(
                **model_parameters
            )
        else:
            raise Exception("模型名称错误！")

        # 开始训练
        start_t = time.time()
        self.model.fit(train_dataset)
        end_t = time.time()
        print(f"训练耗时: {end_t - start_t} 秒")
        pass

    def predict(self, history, predict_window, extra_parameters=None):
        # 转换np数据格式
        history = np.array(history)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            history = self.scaler.transform(history.reshape(-1, 1)).reshape(-1)

        # 先初始化 dataframe
        out_chunk_len = extra_parameters["pred_len"]
        history_df = pd.DataFrame(
            {
                'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
                'value': history,
            },
        )
        history_dataset = TSDataset.load_from_dataframe(
            history_df,  # Also can be path to the CSV file
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )
        predicted_dataset = self.model.predict(history_dataset, )
        pred = predicted_dataset.to_numpy().reshape(-1, ).tolist()
        assert predict_window <= len(pred)
        pred = pred[0:predict_window]

        # 转换
        pred = np.array(pred)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            pred = self.scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

        return pred

    def recursive_predict(self, history, predict_window, extra_parameters=None):
        # 转换np数据格式
        history = np.array(history)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            history = self.scaler.transform(history.reshape(-1, 1)).reshape(-1)
        # 先初始化 dataframe
        out_chunk_len = extra_parameters["pred_len"]
        history_df = pd.DataFrame(
            {
                'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
                'value': history,
            },
        )
        history_dataset = TSDataset.load_from_dataframe(
            history_df,  # Also can be path to the CSV file
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )
        # 滚动预测
        rolling_predicted_dataset = self.model.recursive_predict(tsdataset=history_dataset,
                                                                 predict_length=predict_window)
        pred = rolling_predicted_dataset.to_numpy().reshape(-1, ).tolist()
        assert predict_window <= len(pred)
        pred = pred[0:predict_window]

        # 如果标准化
        if extra_parameters["is_scaler"]:
            pred = self.scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

        return pred


def Test():

    # 用正弦函数生成数据，1000个点，周期为100
    x = np.linspace(0, 100, 1000)
    y = np.sin(x)
    # extra_parameters
    extra_parameters = {
        "seq_len": 128,
        "pred_len": 10,
    }
    # 分别测试不同的模型
    model_names = [
        # "RNNBlockRegressor",
        # "LSTNetRegressor",
        # "MLPRegressor",
        # "NBEATSModel",
        # "TCNRegressor",
        # "NHiTSModel",
        # "TransformerModel",
        "InformerModel",
        "SCINetModel",
    ]
    for model_name in model_names:
        print(f"测试 {model_name} 模型")
        model = PaddleTS_model(name=model_name)
        model.train(history=y, extra_parameters=extra_parameters)
        predict = model.predict(history=y, predict_window=10, extra_parameters=extra_parameters)
        print(f"predict: {predict}")

        # 递归预测
        predict = model.recursive_predict(history=y, predict_window=100, extra_parameters=extra_parameters)
        print(f"predict: {predict}")

        print(f"测试 {model_name} 模型结束")



if __name__ == "__main__":
    Test()
