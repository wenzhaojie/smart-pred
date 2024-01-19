from smart_pred.model.base import Basic_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from paddlets import TSDataset
from paddlets.models.forecasting import RNNBlockRegressor, LSTNetRegressor, MLPRegressor, NBEATSModel, TCNRegressor, NHiTSModel, TransformerModel, InformerModel, DeepARModel
import copy
import time


class NHiTSModel_model(Basic_model):
    def __init__(self, name="NHiTSModel", scaler=StandardScaler()):
        """
        初始化NHiTSModel模型。
        参数:
        - name: 模型的名称，默认为"NHiTSModel"。
        - scaler: 数据标准化处理器，默认使用StandardScaler。
        """
        super().__init__(name, scaler)
        self.name = name
        self.scaler = scaler
        # 设置默认参数
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,  # 输入序列长度
            "pred_len": 1440,  # 预测序列长度
            "history_error_correct": False,  # 是否使用历史误差校正
            "is_scaler": False,  # 是否使用数据标准化
            "use_future": True,  # 是否使用未来数据
            "is_round": False,  # 是否将预测结果四舍五入
            "num_stacks": 3,  # 模型中的堆栈数量
            "num_blocks": 3,  # 每个堆栈中的块数量
            "num_layers": 2,  # 每个块中的层数量
            "max_epochs": 10,  # 训练的最大迭代次数
            "layer_widths": 512,  # 每层的宽度
        }
        print(f"初始化 {self.name}!")

    def train(self, history, extra_parameters=None):
        """
        训练模型。
        参数:
        - history: 训练数据历史序列。
        - extra_parameters: 附加参数。
        """
        history = np.array(history, dtype=np.float32)
        # 初始化训练数据的DataFrame
        train_df = pd.DataFrame({
            'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
            'value': history
        })
        train_dataset = TSDataset.load_from_dataframe(
            train_df,
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )

        # 初始化模型参数
        model_parameters = copy.deepcopy(self.default_extra_parameters)
        if extra_parameters is not None:
            model_parameters.update(extra_parameters)

        # 初始化NHiTS模型
        self.model = NHiTSModel(
            in_chunk_len=model_parameters["seq_len"],
            out_chunk_len=model_parameters["pred_len"],
            max_epochs=model_parameters["max_epochs"],
            num_stacks=model_parameters["num_stacks"],
            num_blocks=model_parameters["num_blocks"],
            num_layers=model_parameters["num_layers"],
            layer_widths=model_parameters["layer_widths"],
        )

        # 开始训练模型
        start_t = time.time()
        self.model.fit(train_dataset)
        # 训练时间记录（此处未使用）

    def predict(self, history, predict_window, extra_parameters=None):
        """
        使用模型进行预测。
        参数:
        - history: 历史数据。
        - predict_window: 预测窗口大小。
        - extra_parameters: 附加参数。
        """
        history = np.array(history, dtype=np.float32)
        # 初始化预测数据的DataFrame
        history_df = pd.DataFrame({
            'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
            'value': history
        })
        history_dataset = TSDataset.load_from_dataframe(
            history_df,
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )
        predicted_dataset = self.model.predict(history_dataset)
        pred = predicted_dataset.to_numpy().reshape(-1, ).tolist()
        assert predict_window <= len(pred)
        pred = pred[0:predict_window]

        return pred

    def recursive_predict(self, history, predict_window):
        """
        使用模型进行递归预测。
        参数:
        - history: 历史数据。
        - predict_window: 预测窗口大小。
        """
        history = np.array(history)
        # 初始化预测数据的DataFrame
        history_df = pd.DataFrame({
            'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
            'value': history
        })
        history_dataset = TSDataset.load_from_dataframe(
            history_df,
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )
        rolling_predicted_dataset = self.model.recursive_predict(
            tsdataset=history_dataset,
            predict_length=predict_window
        )
        pred = rolling_predicted_dataset.to_numpy().reshape(-1, ).tolist()
        assert predict_window <= len(pred)
        pred = pred[0:predict_window]

        return pred



class TestNHiTSModel_model:
    def __init__(self):
        pass

    def test_predict(self):
        print("test_predict")
        model = NHiTSModel_model()
        extra_parameters = {
            "seq_len": 10,
            "pred_len": 10,
            "history_error_correct": False,
            "is_scaler": False,
            "use_future": True,
            "is_round": False,
            "num_stacks": 3,  # Stack数量。
            "num_blocks": 3,  # 构成每个stack的block数量。
            "num_layers": 2,  # 每个block中分叉结构前的全连接层数量。
            "layer_widths": 512,
            # 每个block中全连接层的神经元数量，如果传入list，则list长度必须等于num_stacks，且list中每个元素对应于当前层的神经元数量。如果传入整数，则每个stack中的block中具有相同的神经元数量。
        }
        # 生成一个正弦波，1000个点，振幅为1，周期为100
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        model.train(history=y[0:800], extra_parameters=extra_parameters)

        pred = model.predict(history=y[0:800], predict_window=10)
        print(f"pred:{pred}")

        recursive_pred = model.recursive_predict(history=y[0:800], predict_window=200)
        print(f"recursive_pred:{recursive_pred}")

        rolling_predicted = model.rolling_predict(history=y[0:800], predict_window=200)
        print(f"rolling_predicted:{rolling_predicted}")



class RNNBlockRegressor_model(Basic_model):
    def __init__(self, name="RNNBlockRegressor", scaler=StandardScaler()):
        """
        初始化RNNBlockRegressor模型。
        参数:
        - name: 模型的名称，默认为"RNNBlockRegressor"。
        - scaler: 数据标准化处理器，默认使用StandardScaler。
        """
        super().__init__(name, scaler)
        self.name = name
        self.scaler = scaler
        # 设置默认参数
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,  # 输入序列长度
            "pred_len": 1440,  # 预测序列长度
            "history_error_correct": False,  # 是否使用历史误差校正
            "is_scaler": False,  # 是否使用数据标准化
            "use_future": True,  # 是否使用未来数据
            "is_round": False,  # 是否将预测结果四舍五入
            "hidden_size": 128,  # 隐藏层大小
            "embedding_size": 128,  # 嵌入层大小
            "num_layers_recurrent": 1,  # 循环层的数量
            "max_epochs": 100,  # 训练的最大迭代次数

        }
        print(f"初始化 {self.name}!")

    def train(self, history, extra_parameters=None):
        """
        训练模型。
        参数:
        - history: 训练数据历史序列。
        - extra_parameters: 附加参数。
        """
        history = np.array(history, dtype=np.float32)
        # 初始化训练数据的DataFrame
        train_df = pd.DataFrame({
            'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
            'value': history
        })
        train_dataset = TSDataset.load_from_dataframe(
            train_df,
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )

        # 初始化模型参数
        model_parameters = copy.deepcopy(self.default_extra_parameters)
        if extra_parameters is not None:
            model_parameters.update(extra_parameters)

        # 初始化RNNBlockRegressor模型
        self.model = RNNBlockRegressor(
            in_chunk_len=model_parameters["seq_len"],
            out_chunk_len=model_parameters["pred_len"],
            max_epochs=model_parameters["max_epochs"],
            hidden_size=model_parameters["hidden_size"],
            num_layers_recurrent=model_parameters["num_layers_recurrent"],
        )

        # 开始训练模型
        start_t = time.time()
        self.model.fit(train_dataset)
        # 训练时间记录（此处未使用）

    def predict(self, history, predict_window, extra_parameters=None):
        """
        使用模型进行预测。
        参数:
        - history: 历史数据。
        - predict_window: 预测窗口大小。
        - extra_parameters: 附加参数。
        """
        history = np.array(history, dtype=np.float32)
        # 初始化预测数据的DataFrame
        history_df = pd.DataFrame({
            'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
            'value': history
        })
        history_dataset = TSDataset.load_from_dataframe(
            history_df,
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )
        predicted_dataset = self.model.predict(history_dataset)
        pred = predicted_dataset.to_numpy().reshape(-1, ).tolist()
        assert predict_window <= len(pred)
        pred = pred[0:predict_window]

        return pred


    def recursive_predict(self, history, predict_window):
        """
        使用模型进行递归预测。
        参数:
        - history: 历史数据。
        - predict_window: 预测窗口大小。
        """
        history = np.array(history)
        # 初始化预测数据的DataFrame
        history_df = pd.DataFrame({
            'time_col': pd.date_range('2022-01-01', periods=len(history), freq='1min'),
            'value': history
        })
        history_dataset = TSDataset.load_from_dataframe(
            history_df,
            time_col='time_col',
            target_cols='value',
            freq='1min'
        )
        rolling_predicted_dataset = self.model.recursive_predict(
            tsdataset=history_dataset,
            predict_length=predict_window
        )
        pred = rolling_predicted_dataset.to_numpy().reshape(-1, ).tolist()
        assert predict_window <= len(pred)
        pred = pred[0:predict_window]

        return pred



class TestRNNBlockRegressor_model:
    def __init__(self):
        pass

    def test_predict(self):
        print("test_predict")
        model = RNNBlockRegressor_model()
        extra_parameters = {
            "seq_len": 120,
            "pred_len": 10,
            "history_error_correct": False,
            "is_scaler": False,
            "use_future": True,
            "is_round": False,
            "hidden_size": 128,  # 隐藏层大小
            "embedding_size": 128,  # 嵌入层大小
            "num_layers_recurrent": 1,  # 循环层的数量
        }
        # 生成一个正弦波，1000个点，振幅为1，周期为100
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        model.train(history=y[0:800], extra_parameters=extra_parameters)

        pred = model.predict(history=y[0:800], predict_window=10)
        print(f"pred:{pred}")

        recursive_pred = model.recursive_predict(history=y[0:800], predict_window=200)
        print(f"recursive_pred:{recursive_pred}")

        rolling_predicted = model.rolling_predict(history=y[0:800], predict_window=200, test=y[800:1000], use_future=False)
        print(f"rolling_predicted:{rolling_predicted}")





if __name__ == "__main__":
    # test = TestNHiTSModel_model()
    #
    # test.test_predict()

    test = TestRNNBlockRegressor_model()

    test.test_predict()

