# 导入必要的库和模块
from smart_pred.model.local.base import Basic_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 忽略警告信息
import warnings

warnings.filterwarnings('ignore')

'''
Movingavg_model:
基于一定历史窗口的平均值来预测未来的值
'''


class Movingavg_model(Basic_model):
    def __init__(self, name="Movingavg_model", scaler=MinMaxScaler()):
        print(f"初始化 {name}!")
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1,
            "moving_window": 20,
            "is_scaler": False,
            "is_round": True,
        }

    def predict(self, history, predict_window, extra_parameters=None):
        moving_window = self.default_extra_parameters["moving_window"]
        if extra_parameters and "moving_window" in extra_parameters:
            moving_window = extra_parameters["moving_window"]

        slide = history[-moving_window:]
        predict = np.mean(slide)
        predict_list = [predict for _ in range(predict_window)]
        return np.array(predict_list)


'''
Movingmax_model:
基于一定历史窗口的最大值来预测未来的值
'''


class Movingmax_model(Basic_model):
    def __init__(self, name="Movingmax_model", scaler=MinMaxScaler()):
        print(f"初始化 {name}!")
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1,
            "moving_window": 20,
            "is_scaler": False,
            "is_round": True,
        }

    def predict(self, history, predict_window, extra_parameters=None):
        moving_window = self.default_extra_parameters["moving_window"]
        if extra_parameters and "moving_window" in extra_parameters:
            moving_window = extra_parameters["moving_window"]

        slide = history[-moving_window:]
        predict = max(slide)
        predict_list = [predict for _ in range(predict_window)]
        return np.array(predict_list)


'''
Movingmin_model:
基于一定历史窗口的最小值来预测未来的值
'''


class Movingmin_model(Basic_model):
    def __init__(self, name="Movingmin_model", scaler=MinMaxScaler()):
        print(f"初始化 {name}!")
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1,
            "moving_window": 20,
            "is_scaler": False,
            "is_round": False,
        }

    def predict(self, history, predict_window, extra_parameters=None):
        moving_window = self.default_extra_parameters["moving_window"]
        if extra_parameters and "moving_window" in extra_parameters:
            moving_window = extra_parameters["moving_window"]

        slide = history[-moving_window:]
        predict = min(slide)
        predict_list = [predict for _ in range(predict_window)]
        return np.array(predict_list)
