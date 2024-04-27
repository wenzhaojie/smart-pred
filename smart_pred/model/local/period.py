# 导入必要的库和模块
from smart_pred.model.local.base import Basic_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

'''
Maxvalue_model:
按照天为周期，统计每一时间历史最高值作为预测值
'''
class Maxvalue_model(Basic_model):
    def __init__(self, name="Maxvalue_model", scaler=MinMaxScaler()):
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440*3,
            "pred_len": 1440,
            "is_scaler": False,
            "is_round": False,
        }

    def predict(self, history, predict_window, extra_parameters=None):
        history = np.array(history)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            history = self.scaler.transform(history.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            history = np.round(history)

        num_of_history_day = int(len(history) / 1440)
        history_for_predict = history[-num_of_history_day*1440:]
        predict_list = []
        for index in range(predict_window):
            max_value = max([history_for_predict[i*1440+index] for i in range(num_of_history_day)])
            predict_list.append(max_value)

        predict_list = np.array(predict_list)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            predict_list = self.scaler.inverse_transform(predict_list.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            predict_list = np.round(predict_list)
        return predict_list

'''
Minvalue_model:
按照天为周期，统计每一时间历史最小值作为预测值
'''
class Minvalue_model(Basic_model):
    def __init__(self, name="Minvalue_model", scaler=MinMaxScaler()):
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440*3,
            "pred_len": 1440,
            "is_scaler": False,
            "is_round": False,
        }

    def predict(self, history, predict_window, extra_parameters=None):
        history = np.array(history)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            history = self.scaler.transform(history.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            history = np.round(history)

        num_of_history_day = int(len(history) / 1440)
        history_for_predict = history[-num_of_history_day*1440:]
        predict_list = []
        for index in range(predict_window):
            min_value = min([history_for_predict[i*1440+index] for i in range(num_of_history_day)])
            predict_list.append(min_value)

        predict_list = np.array(predict_list)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            predict_list = self.scaler.inverse_transform(predict_list.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            predict_list = np.round(predict_list)
        return predict_list

'''
Avgvalue_model:
按照天为周期，统计每一时间历史平均值作为预测值
'''
class Avgvalue_model(Basic_model):
    def __init__(self, name="Avgvalue_model", scaler=MinMaxScaler()):
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440*3,
            "pred_len": 1440,
            "is_scaler": False,
            "is_round": False,
        }

    def predict(self, history, predict_window, extra_parameters=None):
        history = np.array(history)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            history = self.scaler.transform(history.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            history = np.round(history)

        num_of_history_day = int(len(history) / 1440)
        history_for_predict = history[-num_of_history_day*1440:]
        predict_list = []
        for index in range(predict_window):
            avg_value = np.mean([history_for_predict[i*1440+index] for i in range(num_of_history_day)])
            predict_list.append(avg_value)

        predict_list = np.array(predict_list)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            predict_list = self.scaler.inverse_transform(predict_list.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            predict_list = np.round(predict_list)
        return predict_list


if __name__ == "__main__":
    my_model = Maxvalue_model()
    history = [1999]+[i*2 for i in range(1439)] + [i*3 for i in range(1440)] + [i*1 for i in range(1440)]
    result = my_model.predict(history, predict_window=1440)
    print(result)

    my_model = Avgvalue_model()
    result = my_model.predict(history, predict_window=1440)
    print(result)
