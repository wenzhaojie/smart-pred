from smart_pred.model.base import Basic_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


class NeuralForecast_model(Basic_model):

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





    def predict(self, history, predict_window, extra_parameters=None):
        """
        预测函数。
        参数:
        - history: 用于预测的历史数据。
        - predict_window: 预测窗口大小。
        - extra_parameters: 附加参数。
        """
        return np.zeros(predict_window)