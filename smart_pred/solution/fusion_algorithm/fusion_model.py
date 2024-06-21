import time

import sklearn

from smart_pred.model.local.base import Basic_model
from smart_pred.model.local.neuralforecast_model import NeuralForecast_model
from smart_pred.model.local.passive import Movingavg_model, Movingmax_model, Movingmin_model
from smart_pred.model.local.period import Maxvalue_model, Minvalue_model, Avgvalue_model
from smart_pred.model.local.quantile import Quantile_model
from smart_pred.utils.metrics import  selective_asymmetric_sample_loss_mae, selective_asymmetric_sample_loss_mse
import numpy as np
from smart_pred.utils.metrics import get_metric_dict


fusion_model_dict = {
    "NHITS": NeuralForecast_model(name="NHITS"),
    "NBEATS": NeuralForecast_model(name="NBEATS"),
    "PatchTST": NeuralForecast_model(name="PatchTST"),
    "DLinear": NeuralForecast_model(name="DLinear"),
    "Movingavg_model": Movingavg_model(name="Movingavg_model"),
    "Movingmax_model": Movingmax_model(name="Movingmax_model"),
    "Movingmin_model": Movingmin_model(name="Movingmin_model"),
    "Maxvalue_model": Maxvalue_model(name="Maxvalue_model"),
    "Minvalue_model": Minvalue_model(name="Minvalue_model"),
    "Avgvalue_model": Avgvalue_model(name="Avgvalue_model"),
    "Quantile_model": Quantile_model(name="Quantile_model"),
}



class Fusion_model(Basic_model):
    def __init__(self,  fusion_model_params_dict, name="Fusion_model"):
        super(Fusion_model, self).__init__(name=name)
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "period_length": 1440,
            "complex_model_list": ["NHITS", "NBEATS", "PatchTST", "DLinear"],
            "simple_model_list": ["Movingavg_model", "Movingmax_model", "Movingmin_model", "Maxvalue_model", "Minvalue_model", "Avgvalue_model", "Quantile_model"],
            "loss": "MSELoss",
            "target_loss": 100,
            "determine_ratio": 0.8,
            "loss_threshold": 0.1,
            "is_complex": True,
            "amp": 1.1,
        }
        self.fusion_model_params_dict = fusion_model_params_dict

    def train(self, history, extra_parameters=None):
        # 训练的过程很简单
        # 首先使用history的最后一个周期的数据作为test_data
        test_data = history[-extra_parameters["period_length"]:]
        # 前面的数据作为train_data
        train_data = history[:-extra_parameters["period_length"]]
        # 然后分别回测不同的模型，先测试简单模型
        self.simple_model_backtest_sample_loss_dict = {} # key为model名称model_name，value是一个列表，记录每一个时间点的样本损失
        for model_name in extra_parameters["simple_model_list"]:
            model = fusion_model_dict[model_name]
            # 获得模型的extra_parameters
            model_extra_parameters = self.fusion_model_params_dict[model_name]


            # 先训练
            model.train(
                history=train_data,
                extra_parameters=model_extra_parameters
            )
            # 再预测
            log_dict, predict = model.use_future_rolling_evaluation(
                train=train_data,
                test=test_data,
                extra_parameters=model_extra_parameters
            )
            print(f"{model_name}的回测结果：{log_dict}")

            # 计算样本损失
            sample_loss = selective_asymmetric_sample_loss_mae(
                y_pred=predict,
                y_true=test_data,
            )
            # 保存至 simple_model_backtest_sample_loss_dict
            self.simple_model_backtest_sample_loss_dict[model_name] = sample_loss

        # 再测试复杂模型
        is_complex = extra_parameters["is_complex"]
        if is_complex:
            print("开始测试复杂模型")
            self.complex_model_backtest_sample_loss_dict = {}
            for model_name in extra_parameters["complex_model_list"]:
                model = fusion_model_dict[model_name]
                # 获得模型的extra_parameters
                model_extra_parameters = self.fusion_model_params_dict[model_name]
                model_extra_parameters["loss"] = extra_parameters["loss"]
                # 先训练
                model.train(
                    history=train_data,
                    extra_parameters=model_extra_parameters
                )
                # 再预测
                log_dict, predict = model.use_future_rolling_evaluation(
                    train=train_data,
                    test=test_data,
                    extra_parameters=model_extra_parameters
                )
                print(f"{model_name}的回测结果：{log_dict}")

                # 计算样本损失
                sample_loss = selective_asymmetric_sample_loss_mae(
                    y_pred=predict,
                    y_true=test_data,
                )
                # 保存至 complex_model_backtest_sample_loss_dict
                self.complex_model_backtest_sample_loss_dict[model_name] = sample_loss

        # 计算融合模型的权重
        # 构造 model_weight_dict
        # 权重的计算方法是：样本损失越小，权重越大
        # 样本损失最小的模型的权重为决定系数 determine_ratio
        # 其他模型的权重按照样本损失的大小来分配
        self.model_weight_dict = {}

        # 创建一个新的字典，汇合简单模型和复杂模型的loss dict
        self.model_backtest_sample_loss_dict = {}
        self.model_backtest_sample_loss_dict.update(self.simple_model_backtest_sample_loss_dict)
        if is_complex:
            self.model_backtest_sample_loss_dict.update(self.complex_model_backtest_sample_loss_dict)



        for model_name in self.model_backtest_sample_loss_dict.keys():
            # 创建一个list用于保存权重
            self.model_weight_dict[model_name] = []

        # 遍历每一个时间点
        for index in range(len(test_data)):
            # 对不同模型，找到对应时间点的样本损失
            sample_loss_list = []
            model_name_list = []
            for model_name in self.model_backtest_sample_loss_dict.keys():
                sample_loss_list.append(self.model_backtest_sample_loss_dict[model_name][index])
                model_name_list.append(model_name)

            # 计算权重
            self.weight_list = []
            # 找到损失最小的模型，计算权重为 determine_ratio
            determine_ratio = extra_parameters["determine_ratio"]
            min_index = sample_loss_list.index(min(sample_loss_list))
            for i in range(len(sample_loss_list)):
                if i == min_index:
                    self.weight_list.append(determine_ratio)
                    # min loss is
                    print(f"min loss: {sample_loss_list[min_index]}, model: {model_name_list[min_index]}")
                    print(f"min loss value: {test_data[index]}")
                else:
                    self.weight_list.append((1 - determine_ratio) * sample_loss_list[min_index] / sample_loss_list[i])
            # 保存权重
            for i, model_name in enumerate(model_name_list):
                self.model_weight_dict[model_name].append(self.weight_list[i])


    def predict(self, history, predict_window, extra_parameters=None):
        # 预测的过程也很简单
        # 对不同model进行预测
        model_predict_list_dict = {}
        for model_name in self.model_backtest_sample_loss_dict.keys():
            model = fusion_model_dict[model_name]
            # 获得模型的extra_parameters
            model_extra_parameters = self.fusion_model_params_dict[model_name]
            model_extra_parameters["loss"] = extra_parameters["loss"]
            # 先训练
            model.train(
                history=history,
                extra_parameters=model_extra_parameters
            )
            # 再预测
            predict = model.predict(
                history=history,
                predict_window=predict_window,
                extra_parameters=model_extra_parameters
            )
            model_predict_list_dict[model_name] = predict

        # 对不同模型的预测结果进行加权平均
        predict_list = []
        for i in range(predict_window):
            predict = 0
            for model_name in model_predict_list_dict.keys():
                predict += model_predict_list_dict[model_name][i] * self.model_weight_dict[model_name][i]
                # amp
                predict = predict * extra_parameters["amp"]
            predict_list.append(predict)

        return predict_list

    def use_future_rolling_evaluation(self, train, test, extra_parameters=None):
        if extra_parameters is None:
            extra_parameters = self.default_extra_parameters
        else:
            for key, value in self.default_extra_parameters.items():
                if key not in extra_parameters:
                    extra_parameters[key] = value
        # 先转换np
        train = np.array(train)
        test = np.array(test)

        # 开始计时
        start_t = time.time()

        # 对不同model进行预测
        model_predict_list_dict = {}
        print(f"model_backtest_sample_loss_dict.keys():{self.model_backtest_sample_loss_dict.keys()}")
        for model_name in self.model_backtest_sample_loss_dict.keys():
            model = fusion_model_dict[model_name]
            # 获得模型的extra_parameters
            model_extra_parameters = self.fusion_model_params_dict[model_name]
            model_extra_parameters["loss"] = extra_parameters["loss"]
            # 先训练
            model.train(
                history=train,
                extra_parameters=model_extra_parameters
            )
            # 再预测
            log_dict, predict = model.use_future_rolling_evaluation(
                train=train,
                test=test,
                extra_parameters=model_extra_parameters
            )
            model_predict_list_dict[model_name] = predict

        # 对不同模型的预测结果进行加权平均
        predict_list = []
        for i in range(len(test)):
            predict = 0
            for model_name in model_predict_list_dict.keys():
                predict += model_predict_list_dict[model_name][i] * self.model_weight_dict[model_name][i]
                # amp
                predict = predict * extra_parameters["amp"]
            predict_list.append(predict)

        # 如果预测的长度超过了test的长度，需要截断
        predict = predict_list[:len(test)]

        print(f"len(predict):{len(predict)}")

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

        

        









