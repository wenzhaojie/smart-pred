import copy
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from smart_pred.utils.metrics import get_metric_dict

class Basic_model:
    def __init__(self, name="Basic_model", scaler=StandardScaler()):
        """
        模型初始化函数。
        参数:
        - name: 模型的名字，默认为"Basic_model"。
        - scaler: 数据的标准化处理器，默认使用StandardScaler。
        """
        self.scaler = scaler # 关于 scaler，只在train的时候执行fit_transform，predict和rolling predict的时候执行transform即可
        self.name = name
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "moving_window": 20,
            "is_scaler": False,
            "is_round": False,
            "period_length": 1440,
            "start_at_period": 0,
            "amplify": 1.0
        }

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

    def train(self, history, extra_parameters=None):
        """
        训练模型的函数。
        参数:
        - history: 训练数据。
        - extra_parameters: 附加参数。
        """
        # 转换np
        history = np.array(history)
        # 如果标准化了
        if extra_parameters["is_scaler"]:
            history = self.scaler.fit_transform(history.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            history = np.round(history)

        pass

    def predict(self, history, predict_window, extra_parameters=None):
        """
        预测函数。
        参数:
        - history: 用于预测的历史数据。
        - predict_window: 预测窗口大小。
        - extra_parameters: 附加参数。
        """
        return np.zeros(predict_window)

    def rolling_predict(self, history, predict_window, extra_parameters=None):
        """
        滚动预测函数。
        参数:
        - history: 用于预测的历史数据。
        - predict_window: 预测窗口大小。
        - extra_parameters: 附加参数。
        """
        # 如果没有提供额外参数，则使用默认参数
        if extra_parameters is None:
            extra_parameters = self.default_extra_parameters
        else:
            # 如果提供了额外参数，则使用默认参数补充额外参数
            for key, value in self.default_extra_parameters.items():
                if key not in extra_parameters:
                    extra_parameters[key] = value

        seq_len = extra_parameters["seq_len"]
        pred_len = extra_parameters["pred_len"]
        period_length = extra_parameters["period_length"]

        # is_scaler为True时，对数据进行标准化处理
        if extra_parameters["is_scaler"]:
            history = self.scaler.transform(history.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            history = np.round(history)

        # 转换历史数据为列表格式
        history = list(history)

        predict_list = []  # 存储预测结果
        pointer = 0  # 指针，用于遍历历史数据

        history_add = predict_list
        while len(predict_list) < predict_window:
            # 获取输入数据
            if pointer < seq_len:
                history_base = history[-seq_len + pointer:]
                train = history_base + history_add[:pointer]
            else:
                history_base = []
                train = history_base + history_add[pointer - seq_len:pointer]

            start_at_period = int(pointer) % period_length
            extra_parameters["start_at_period"] = start_at_period

            assert len(train) == seq_len
            predict = self.predict(history=train, predict_window=pred_len, extra_parameters=extra_parameters)
            assert len(predict) == pred_len
            predict_list.extend(predict)
            pointer += pred_len

        predict = predict_list
        # 最后收尾阶段
        # 有可能预测超出了我们所需要的范围，所以需要截断
        rolling_predict = predict[:predict_window]
        rolling_predict = np.array(rolling_predict)
        # 如果需要标准化处理，则进行逆标准化处理
        if extra_parameters["is_scaler"]:
            rolling_predict = self.scaler.inverse_transform(rolling_predict.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            rolling_predict = np.round(rolling_predict)

        return rolling_predict


    def use_future_rolling_evaluation(self, train, test, extra_parameters=None):
        """
        模型评估函数。
        参数:
        - train: 训练数据。
        - test: 测试数据。
        - extra_parameters: 附加参数。
        """

        if extra_parameters is None:
            extra_parameters = self.default_extra_parameters
        else:
            for key, value in self.default_extra_parameters.items():
                if key not in extra_parameters:
                    extra_parameters[key] = value
        # 先转换np
        train = np.array(train)
        test = np.array(test)

        # 如果需要标准化处理，则进行标准化处理
        if extra_parameters["is_scaler"]:
            train = self.scaler.transform(train.reshape(-1, 1)).reshape(-1)
            test = self.scaler.transform(test.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            train = np.round(train)
            test = np.round(test)

        # 滚动预测，每一次使用真实的数据作为模型的输入
        start_t = time.time()
        predict = []
        # 每一次pred_len
        pred_len = extra_parameters["pred_len"]
        while len(predict) < len(test):
            # history有两部分组成
            # 1. train的部分
            # 2. test的部分
            history = np.concatenate((train, test[:len(predict)]))
            # 只取最后seq_len个数据
            history = history[-extra_parameters["seq_len"]:]
            partial_predict = self.predict(
                history=history,
                predict_window=pred_len,
                extra_parameters=extra_parameters
            )
            predict.extend(partial_predict)
        # 如果预测的长度超过了test的长度，需要截断
        predict = predict[:len(test)]

        print(f"len(predict):{len(predict)}")

        # predict amplify
        if "amplify" in extra_parameters.keys():
            amplify = extra_parameters["amplify"]
            print(f"放大系数: {amplify}")
            predict = [item * amplify for item in predict]
            predict = np.array(predict)

        # 计算评估指标
        metric_dict = get_metric_dict(test, predict)
        _cold_start_invocation_ratio = metric_dict["cold_start_invocation_ratio"]
        _utilization_ratio = metric_dict["utilization_ratio"]
        _over_provisioned_ratio = metric_dict["over_provisioned_ratio"]
        _cold_start_time_slot_ratio = metric_dict["cold_start_time_slot_ratio"]

        # 如果归一化
        if extra_parameters["is_scaler"]:
            predict = self.scaler.transform(np.array(predict).reshape(-1, 1)).reshape(-1)

        # 计算评估指标
        metric_dict = get_metric_dict(test, predict)
        # 覆盖原有的评估指标
        metric_dict["cold_start_invocation_ratio"] = _cold_start_invocation_ratio
        metric_dict["utilization_ratio"] = _utilization_ratio
        metric_dict["over_provisioned_ratio"] = _over_provisioned_ratio
        metric_dict["cold_start_time_slot_ratio"] = _cold_start_time_slot_ratio

        print("predict:", predict)

        # 转换成ndarray
        predict = np.array(predict)
        predict_t = time.time() - start_t

        # 收集日志
        log_dict = {
            "model": self.name,
            "train_length": len(train),
            "test_length": len(test),
            "predict_t": predict_t,
        }
        log_dict.update(extra_parameters)
        log_dict.update(metric_dict)
        print(f"log:{log_dict}")

        # 如果需要标准化处理，则进行逆标准化处理
        if extra_parameters["is_scaler"]:
            predict = self.scaler.inverse_transform(predict.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            predict = np.round(predict)

        return log_dict, predict




 
