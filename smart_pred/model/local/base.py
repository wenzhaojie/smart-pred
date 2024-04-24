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
        return np.array(rolling_predict)

    def evaluate(self, train, test, extra_parameters=None):
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

        # 是否四舍五入
        is_round = extra_parameters["is_round"]
        # 是否归一化
        is_scaler = extra_parameters["is_scaler"]

        # 先做归一化
        if is_scaler == True:
            combined_trace = np.concatenate((np.array(train), np.array(test)))
            processed_trace = self.scaler.fit_transform(combined_trace.reshape(-1, 1)).reshape(-1, )
            train = processed_trace[:len(train)]
            test = processed_trace[len(train):]
        else:
            train = np.array(train)
            test = np.array(test)

        # 再训练
        start_t = time.time()
        self.train(history=train, extra_parameters=extra_parameters)
        train_t = time.time() - start_t

        # 滚动预测
        start_t = time.time()
        predict = self.rolling_predict(
            history=train,
            predict_window=len(test),
            extra_parameters=extra_parameters
        )
        predict_t = time.time() - start_t  # 计算时间

        # 做还原归一化
        if is_scaler == True:
            predict = self.scaler.inverse_transform(predict.reshape(-1, 1)).reshape(-1, )
            test = self.scaler.inverse_transform(test.reshape(-1, 1)).reshape(-1, )

        # 如果四舍五入
        if is_round:
            round_predict = []
            for num in predict:
                round_predict.append(max(0, round(num)))
            predict = round_predict

        # 指标计算
        metrics_dict = get_metric_dict(y_pred=predict, y_test=test)

        # 收集日志
        log_dict = {
            "model": self.name,
            "train_length": len(train),
            "test_length": len(test),
            "predict_t": predict_t,
            "train_t": train_t
        }
        log_dict.update(extra_parameters)
        log_dict.update(metrics_dict)
        print(f"log:{log_dict}")

        return log_dict


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

        # 是否四舍五入
        is_round = extra_parameters["is_round"]
        # 是否归一化
        is_scaler = extra_parameters["is_scaler"]

        # 先做归一化
        if is_scaler == True:
            combined_trace = np.concatenate((np.array(train), np.array(test)))
            processed_trace = self.scaler.fit_transform(combined_trace.reshape(-1, 1)).reshape(-1, )
            train = processed_trace[:len(train)]
            test = processed_trace[len(train):]
        else:
            train = np.array(train)
            test = np.array(test)

        # 再训练
        start_t = time.time()
        self.train(history=train, extra_parameters=extra_parameters)
        train_t = time.time() - start_t

        # 滚动预测，每一次使用真实的数据作为模型的输入
        start_t = time.time()
        predict = []
        # 每一次pred_len
        pred_len = extra_parameters["pred_len"]
        for i in range(len(test)):
            partial_predict = self.predict(
                history=np.concatenate((train, test[:i])),
                predict_window=pred_len,
                extra_parameters=extra_parameters
            )
            predict.extend(partial_predict)
        # 转换成ndarray
        predict = np.array(predict)
        predict_t = time.time() - start_t

        # 做还原归一化
        if is_scaler == True:
            predict = self.scaler.inverse_transform(predict.reshape(-1, 1)).reshape(-1, )
            test = self.scaler.inverse_transform(test.reshape(-1, 1)).reshape(-1, )

        # 如果四舍五入
        if is_round:
            round_predict = []
            for num in predict:
                round_predict.append(max(0, round(num)))
            predict = round_predict

        # 指标计算
        metrics_dict = get_metric_dict(y_pred=predict, y_test=test)

        # 收集日志
        log_dict = {
            "model": self.name,
            "train_length": len(train),
            "test_length": len(test),
            "predict_t": predict_t,
            "train_t": train_t
        }
        log_dict.update(extra_parameters)
        log_dict.update(metrics_dict)
        print(f"log:{log_dict}")

        return log_dict




 
