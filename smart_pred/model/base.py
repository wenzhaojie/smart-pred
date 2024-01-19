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
            "history_error_correct": False,
            "is_scaler": False,
            "use_future": True,
            "is_round": False,
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

    def rolling_predict(self, history, predict_window, test=None, use_future=True, extra_parameters=None):
        """
        滚动预测函数。
        参数:
        - history: 用于预测的历史数据。
        - predict_window: 预测窗口大小。
        - test: 用于评估预测性能的测试数据。
        - use_future: 是否使用未来数据。
        - extra_parameters: 附加参数。
        """
        # 如果没有提供额外参数，则使用默认参数
        if extra_parameters is None:
            extra_parameters = self.default_extra_parameters

        try:
            seq_len = extra_parameters["seq_len"]
            pred_len = extra_parameters["pred_len"]
        except Exception as e:
            seq_len = self.default_extra_parameters["seq_len"]
            pred_len = self.default_extra_parameters["pred_len"]
            print(e)

        # 转换历史数据为列表格式
        history = list(history)

        predict_list = []  # 存储预测结果
        compute_t_list = []  # 存储计算时间
        pointer = 0  # 指针，用于遍历历史数据

        # 判断是否使用测试数据
        if use_future:
            assert len(test) == predict_window
            history_add = list(test)
        else:
            history_add = predict_list

        while len(predict_list) < predict_window:
            # 获取输入数据
            if pointer < seq_len:
                history_base = history[-seq_len + pointer:]
                train = history_base + history_add[:pointer]
            else:
                history_base = []
                train = history_base + history_add[pointer - seq_len:pointer]

            start_at_period = int(pointer) % 1440
            extra_parameters["start_at_period"] = start_at_period

            assert len(train) == seq_len
            predict = self.predict(history=train, predict_window=pred_len, extra_parameters=extra_parameters)
            assert len(predict) == pred_len
            predict_list.extend(predict)
            pointer += pred_len

        # 使用历史误差进行预测结果修正
        history_error_correct = extra_parameters["history_error_correct"]
        predict = predict_list
        max_error = 1
        history_error = 1


        if history_error_correct:
            corrected_predict = []
            for index, (pred, true) in enumerate(zip(predict, test)):
                # 修正预测值
                if index == 0:  # 最开始不做修正
                    corrected_value = pred
                    if pred < 1:  # 如果预测值小于1，不做修正
                        history_error = 1
                    else:
                        history_error = float(true / pred)
                        if history_error < 1 - max_error:
                            history_error = 1 - max_error
                        elif history_error > 1 + max_error:
                            history_error = 1 + max_error
                        else:
                            pass

                else:
                    corrected_value = history_error * pred
                    if pred < 1:
                        history_error = 1
                    else:
                        history_error = float(true / pred)
                        if history_error < 1 - max_error:
                            history_error = 1 - max_error
                        elif history_error > 1 + max_error:
                            history_error = 1 + max_error
                        else:
                            pass
                corrected_predict.append(corrected_value)
            predict = np.array(corrected_predict)

        # 最后收尾阶段
        # 有可能预测超出了我们所需要的范围，所以需要截断
        rolling_predict = predict[:predict_window]
        return np.array(rolling_predict)

    def evaluate(self, train, test, extra_parameters=None, plotter=None):
        """
        模型评估函数。
        参数:
        - train: 训练数据。
        - test: 测试数据。
        - extra_parameters: 附加参数。
        - plotter: 绘图工具。
        """
        use_future = extra_parameters["use_future"]
        is_round = extra_parameters["is_round"]
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
            test=test,
            use_future=use_future,
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

        # 画图
        if plotter != None:
            print(f"开始画图")
            plotter.plot_lines(
                y_list=[predict, test],
                y_label_list=["Predict", "Real"],
                x_label="Timestamp (min)",
                y_label="Concurrency",
                title=plotter.title,
                save_root=plotter.save_root,
                filename=f"Predict vs Real {self.name}",
            )
            for i in range(24):
                plotter.plot_lines(
                    y_list=[predict[i * 60:(i + 1) * 60], test[i * 60:(i + 1) * 60]],
                    y_label_list=["Predict", "Real"],
                    x_label="Timestamp (min)",
                    y_label="Concurrency",
                    title=plotter.title,
                    save_root=plotter.save_root,
                    filename=f"Evaluation {i}th Hour {self.name}",
                )

        return log_dict, predict