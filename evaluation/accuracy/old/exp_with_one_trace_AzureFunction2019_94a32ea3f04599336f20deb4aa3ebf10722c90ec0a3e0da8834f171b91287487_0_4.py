import time

import numpy as np

from smart_pred.model.local.histogram import Histogram_model, OpenFaaS_model
from smart_pred.model.local.fourier import Crane_dsp_model, Icebreaker_model

from smart_pred.solution.fusion_algorithm.fusion_model_segment import Fusion_segment_model
from smart_pred.solution.fusion_algorithm.fusion_model_segment_bursty import Fusion_segment_bursty_model
from smart_pred.model.local.period import Maxvalue_model, Minvalue_model, Avgvalue_model
from smart_pred.model.local.passive import Movingavg_model, Movingmax_model, Movingmin_model
from smart_pred.model.local.quantile import Quantile_model

from smart_pred.model.local.neuralforecast_model import NeuralForecast_model

from smart_pred.dataset.crane_trace import CraneDataset
from smart_pred.dataset.huawei import HuaweiPrivateDataset, HuaweiPublicDataset
from smart_pred.dataset.azure_trace_2019 import AzureFunction2019
from sklearn.preprocessing import StandardScaler


from smart_pred.utils.metrics import cold_start_invocation_ratio, utilization_ratio, over_provisioned_ratio, cold_start_time_slot_ratio
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from smart_pred.utils.nf_loss import SelectiveAsymmetricMAELoss, SelectiveAsymmetricMSELoss

from py_plotter.plot import Plotter
import torch


class ExpOneTrace:

    # 准备好测试一个trace所需的所有东西
    # 包括模型对象，模型超参数配置
    def __init__(self, dataset_name="AzureFunction2019", function_name="94a32ea3f04599336f20deb4aa3ebf10722c90ec0a3e0da8834f171b91287487", start_day=0, end_day=4):
        self.dataset_name = dataset_name
        self.function_name = function_name
        self.start_day = start_day
        self.end_day = end_day
        self.scaled_test_data = None
        self.scaled_data = None
        self.scaler = None
        self.baselines_dict = None
        self.train_data = None
        self.test_data = None
        self.init_trace(dataset_name, function_name, start_day, end_day)
        self.init_baselines()

    def init_trace(self, dataset_name, function_name, start_day=0, end_day=4):
        if dataset_name == "CraneDataset":
            dataset = CraneDataset()
            data = dataset.get_data_by_day_range(
                function_name=str(function_name),
                start_day=start_day,
                end_day=5,
                resolution="minute",
                data_type="requests",
            )

        elif dataset_name == "HuaweiPrivateDataset":
            dataset = HuaweiPrivateDataset()
            data = dataset.get_data_by_day_range(
                function_name=str(function_name),
                start_day=start_day,
                end_day=end_day,
                resolution="minute",
                data_type="requests",
            )

        elif dataset_name == "HuaweiPublicDataset":
            dataset = HuaweiPublicDataset()
            data = dataset.get_data_by_day_range(
                function_name=str(function_name),
                start_day=start_day,
                end_day=end_day,
                resolution="minute",
                data_type="requests",
            )

        elif dataset_name == "AzureFunction2019":
            dataset = AzureFunction2019()
            dataset.load_and_cache_dataset()
            data = dataset.get_all_invocation_trace_by_hash_function(
                hash_function=function_name
            )
            # 一共需要多长
            length = 1440 * (end_day - start_day + 1)
            data = data[:length]

        else:
            raise ValueError(f"dataset_name:{dataset_name} is not supported.")

        # 处理NaN
        data = np.array(data)
        for i in range(len(data)):
            if data[i] != data[i]:
                data[i] = 0

        # data训练归一化scaler
        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
        self.scaler = scaler
        self.scaled_data = scaled_data
        self.scaled_test_data = scaled_data[-1440:]

        train_data = data[:-1440]
        test_data = data[-1440:]

        self.train_data = train_data
        self.test_data = test_data

    def init_baselines(self):
        self.baselines_dict = {
            "OpenFaaS": Movingmax_model(),
            "Crane": Crane_dsp_model(),
            "Icebreaker": Icebreaker_model(),
            "Histogram": Movingavg_model(),
            "SmartPred": Fusion_segment_bursty_model(),
            "NHITS": NeuralForecast_model(name="NHITS"),
            "NBEATS": NeuralForecast_model(name="NBEATS"),
            "PatchTST": NeuralForecast_model(name="PatchTST"),
            "DLinear": NeuralForecast_model(name="DLinear"),
            # "Movingavg": Movingavg_model(),
            # "Movingmax_model": Movingmax_model(),
            # "Movingmin_model": Movingmin_model(),
        }

        # 初始化 fusion_model的参数
        self.baselines_dict["SmartPred"].init_fusion_model_params_dict(
            {
                "NHITS": {
                    "seq_len": 1440,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                    "period": 1440,
                    "max_steps": 100,
                },
                "NBEATS": {
                    "seq_len": 1440,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                    "period": 1440,
                    "max_steps": 100,
                },
                "PatchTST": {
                    "seq_len": 1440,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                    "period": 1440,
                    "max_steps": 100,
                },
                "DLinear": {
                    "seq_len": 1440,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                    "period": 1440,
                    "max_steps": 100,
                },
                "Movingavg_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1,
                    "moving_window": 20,
                    "is_scaler": False,
                    "is_round": True,
                    "max_steps": 100,
                },
                "Movingmax_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1,
                    "moving_window": 20,
                    "is_scaler": False,
                    "is_round": True,
                    "max_steps": 100,
                },
                "Movingmin_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1,
                    "moving_window": 20,
                    "is_scaler": False,
                    "is_round": True,
                },
                "Maxvalue_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                },
                "Minvalue_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                },
                "Avgvalue_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                },
                "Quantile_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                    "quantile": 0.5,
                },
                "Histogram_model": {
                    "seq_len": 1440 * 3,
                    "pred_len": 1440,
                    "is_scaler": False,
                    "is_round": False,
                },
            }
        )

        self.baselines_config_dict = {
            "NHITS": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "NBEATS": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "PatchTST": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "DLinear": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "Movingavg": {
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "moving_window": 40,
                "is_scaler": False,
                "is_round": False,
                "max_steps": 100,
                "amplify": 1.0,
            },
            "Movingmax": {
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "moving_window": 20,
                "is_scaler": False,
                "is_round": False,
                "max_steps": 100,
                "amplify": 1.0,
            },
            "Movingmin": {
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "moving_window": 20,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.0,
            },
            "Maxvalue": {
                "seq_len": 1440 * 3,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.0,
            },
            "OpenFaaS": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "amplify": 1.0,
                "moving_window": 10,
            },
            "Crane": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 3,
                "pred_len": 1440,
                "amplify": 1
            },
            "Icebreaker": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 5,
                "pred_len": 1440,
            },
            "Histogram": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "amplify": 1.0,
                "moving_window": 10,
            },
            "SmartPred": {
                "seq_len": 1440 * 4,
                "pred_len": 1440,
                "period_length": 1440,
                "complex_model_list": ["NHITS", "NBEATS", "PatchTST", "DLinear"],
                "simple_model_list": ["Movingmax_model", "Movingavg_model", "Maxvalue_model", "Histogram_model"],
                "loss": "SelectiveAsymmetricMAELoss",
                "determine_ratio": 1,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.00,
                "is_complex": False,
                "delta_t": 120,
                "epsilon": 1.5,
                "bursty_quantile": 1,
            },
        }

    def plot_predict(self, pred, true, save_root="./plot_predict", filename="pred_true.pdf"):
        x = [i for i in range(len(true))]

        my_plotter = Plotter(
            figsize=(10, 8),
            dpi=300,
            font_thirdparty="YaHei",
        )

        my_plotter.plot_lines(
            x_list=[x, x],
            line_data_list=[true, pred],
            legend_label_list=["true", "pred"],
            filename=filename,
            x_label="时间戳（min）",
            y_label="负载水平",
            legend_title=None,
            save_root=save_root,
            x_tick_ndigits=0,
            y_tick_ndigits=0,
        )
        print(f"save plot to {save_root}/{filename}")



    def evaluation(self):
        # 保存每个模型的评估结果
        evaluation_result_dict = {}
        evaluation_metric_dict = {}

        for baseline_name, baseline_model in self.baselines_dict.items():
            print(f"现在测试模型:{baseline_name}")
            baseline_config = self.baselines_config_dict[baseline_name]

            # 先训练
            start_train_t = time.time()
            baseline_model.train(
                history=self.train_data,
                extra_parameters=baseline_config
            )
            end_train_t = time.time()
            train_t = end_train_t - start_train_t
            print(f"训练时间:{train_t}")

            # 在测试
            start_predict_t = time.time()
            _, predict_result = baseline_model.use_future_rolling_evaluation(
                train=self.train_data,
                test=self.test_data,
                extra_parameters=baseline_config
            )
            end_predict_t = time.time()
            predict_t = end_predict_t - start_predict_t

            # predict_result 中小于0 的值全部置为0
            predict_result = np.array(predict_result)
            predict_result[predict_result < 0] = 0

            # 一些指标是不能归一化的，例如 cold_start_invocation_ratio, utilization_ratio, over_provisioned_ratio, cold_start_time_slot_ratio
            _cold_start_invocation_ratio = cold_start_invocation_ratio(y_pred=predict_result, y_test=self.test_data)
            _utilization_ratio = utilization_ratio(y_pred=predict_result, y_test=self.test_data)
            _over_provisioned_ratio = over_provisioned_ratio(y_pred=predict_result, y_test=self.test_data)
            _cold_start_time_slot_ratio = cold_start_time_slot_ratio(y_pred=predict_result, y_test=self.test_data)

            # 一些指标可以归一化，例如 MAE, RMSE, MAPE，因此进行归一化处理
            scaled_pred_data = self.scaler.transform(np.array(predict_result).reshape(-1, 1)).reshape(-1)
            scaled_test_data = self.scaled_test_data


            _mae = mean_absolute_error(y_pred=scaled_pred_data, y_true=scaled_test_data)
            _rmse = mean_squared_error(y_pred=scaled_pred_data, y_true=scaled_test_data)
            _mape = mean_absolute_percentage_error(y_pred=scaled_pred_data, y_true=scaled_test_data)
            # 自定义loss
            # 转换torch.tensor
            torch_tensor_pred = torch.tensor(scaled_pred_data).float()
            torch_tensor_test = torch.tensor(scaled_test_data).float()

            _selective_asymmetric_mae = SelectiveAsymmetricMAELoss()(torch_tensor_pred, torch_tensor_test)
            _selective_asymmetric_mse = SelectiveAsymmetricMSELoss()(torch_tensor_pred, torch_tensor_test)

            # 转换成np，取values[0]
            _selective_asymmetric_mae = float(_selective_asymmetric_mae.detach().numpy().item())
            _selective_asymmetric_mse = float(_selective_asymmetric_mse.detach().numpy().item())
            print(f"_selective_asymmetric_mae:{_selective_asymmetric_mae}")
            print(f"_selective_asymmetric_mse:{_selective_asymmetric_mse}")
            print(f"type(_selective_asymmetric_mae):{type(_selective_asymmetric_mae)}")
            print(f"type(_selective_asymmetric_mse):{type(_selective_asymmetric_mse)}")

            # 生成评估指标字典
            metric_dict = {
                "baseline_name": baseline_name,
                "cold_start_invocation_ratio": _cold_start_invocation_ratio,
                "utilization_ratio": _utilization_ratio,
                "over_provisioned_ratio": _over_provisioned_ratio,
                "cold_start_time_slot_ratio": _cold_start_time_slot_ratio,
                "mae": _mae,
                "rmse": _rmse,
                "mape": _mape,
                "selective_asymmetric_mae": _selective_asymmetric_mae,
                "selective_asymmetric_mse": _selective_asymmetric_mse,
                "train_t": train_t,
                "predict_t": predict_t,
            }
            print(metric_dict)
            # 保存评估指标
            evaluation_metric_dict[baseline_name] = metric_dict

            # 保存预测结果
            evaluation_result_dict[baseline_name] = predict_result

        # 将评估结果保存到json
        # 先构造实验参数字典
        experiment_parameters_dict = {
            "dataset_name": self.dataset_name,
            "function_name": self.function_name,
            "start_day": self.start_day,
            "end_day": self.end_day,
        }
        # 扩展实验参数字典，将evaluation_metric_dict合并进去
        experiment_parameters_dict.update(evaluation_metric_dict)
        # 保存实验参数字典
        # 为当前实验创建一个文件夹
        save_root = f"{self.dataset_name}_{self.function_name}_{self.start_day}_{self.end_day}"
        save_json_path = f"{save_root}/evaluation_baselines_result.json"
        # 创建文件夹
        import os
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)

        import json
        with open(save_json_path, "w") as f:
            json.dump(experiment_parameters_dict, f)

        # 将不同baseline的预测结果导出numpy
        save_npy_root = f"{save_root}/predict_result"
        if not os.path.exists(save_npy_root):
            os.makedirs(save_npy_root, exist_ok=True)

        for baseline_name, predict_result in evaluation_result_dict.items():
            save_npy_path = f"{save_npy_root}/{baseline_name}.npy"
            np.save(save_npy_path, predict_result)

        # 画图
        for baseline_name, predict_result in evaluation_result_dict.items():
            self.plot_predict(
                pred=predict_result,
                true=self.test_data,
                save_root=save_root,
                filename=f"{baseline_name}_pred_true.pdf"
            )



if __name__ == '__main__':
    exp = ExpOneTrace()
    exp.evaluation()
