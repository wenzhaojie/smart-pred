import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from smart_pred.utils.metrics import get_metric_dict

class Basic_model:
    def __init__(self, name="Basic_model", scaler=StandardScaler(), model_parameters=None):
        """
        模型初始化函数。
        参数:
        - name: 模型的名字，默认为"Basic_model"。
        - scaler: 数据的标准化处理器，默认使用StandardScaler。
        """
        self.scaler = scaler
        self.name = name
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "moving_window": 20,
            "is_scaler": False,
            "is_round": False,
            "period_length": 1440,
            "start_at_period": 0,
        }
        self.model_parameters = model_parameters

    def get_scaler(self):
        """
        获取当前模型使用的数据标准化处理器。
        """
        print(f"self.scaler:{self.scaler}")
        return self.scaler

    def get_name(self):
        """
        获取当前模型的名称。
        """
        print(f"self.name:{self.name}")

    def train(self, train_df, extra_parameters=None):
        """
        训练模型的函数。
        参数:
        - train_df: 训练数据。
        - extra_parameters: 附加参数。
        其中 train_df 至少包含三列，分别是：
        Y_df = pd.DataFrame({'y': history, 'ds': pd.date_range(start='2000-01-01', periods=len(history), freq='H')})
        Y_df['unique_id'] = 'time_series' # 这个可以是不同的时间序列的id
        """
        pass

    def predict(self, history_df, predict_window, extra_parameters=None):
        """
        预测函数。
        参数:
        - history_df: 用于预测的历史数据。
        - predict_window: 预测窗口大小，也就是输出多少个预测值。
        - extra_parameters: 附加参数。
        其中 history_df 至少包含三列，分别是：
        Y_df = pd.DataFrame({'y': history, 'ds': pd.date_range(start='2000-01-01', periods=len(history), freq='H')})
        Y_df['unique_id'] = 'time_series' # 这个可以是不同的时间序列的id
        """
        pass

