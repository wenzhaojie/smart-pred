from smart_pred.model.local.base import Basic_model
from smart_pred.model.local.neuralforecast_model import NeuralForecast_model
from smart_pred.model.local.passive import Movingavg_model, Movingmax_model, Movingmin_model
from smart_pred.model.local.period import Maxvalue_model, Minvalue_model, Avgvalue_model
from smart_pred.model.local.quantile import Quantile_model
from smart_pred.utils.metrics import  selective_asymmetric_sample_loss_mae, selective_asymmetric_sample_loss_mse


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

fusion_model_params_dict = {
    "NHITS": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
    },
    "NBEATS": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
    },
    "PatchTST": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
    },
    "DLinear": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
    },
    "Movingavg_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 20,
        "is_scaler": False,
        "is_round": True,
    },
    "Movingmax_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 20,
        "is_scaler": False,
        "is_round": True,
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
}



class Fusion_model(Basic_model):
    def __init__(self, name="Fusion_model"):
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
        }

    def train(self, history, extra_parameters=None):
        # 训练的过程很简单
        # 首先使用history的最后一个周期的数据作为test_data
        test_data = history[-extra_parameters["period_length"]:]
        # 前面的数据作为train_data
        train_data = history[:-extra_parameters["period_length"]]
        # 然后分别回测不同的模型，先测试简单模型
        simple_model_backtest_sample_loss_dict = {} # key为model名称model_name，value是一个列表，记录每一个时间点的样本损失
        for model_name in extra_parameters["simple_model_list"]:
            model = fusion_model_dict[model_name]
            # 获得模型的extra_parameters
            model_extra_parameters = fusion_model_params_dict[model_name]
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
            sample_loss = selective_asymmetric_sample_loss_mse(
                y_pred=predict,
                y_true=test_data,
            )
            # 保存至 simple_model_backtest_sample_loss_dict
            simple_model_backtest_sample_loss_dict[model_name] = sample_loss

        # 再测试复杂模型
        is_complex = False
        if is_complex:
            print("开始测试复杂模型")
            self.complex_model_backtest_sample_loss_dict = {}
            for model_name in extra_parameters["complex_model_list"]:
                model = fusion_model_dict[model_name]
                # 获得模型的extra_parameters
                model_extra_parameters = fusion_model_params_dict[model_name]
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
                sample_loss = selective_asymmetric_sample_loss_mse(
                    y_pred=predict,
                    y_true=test_data,
                )
                # 保存至 complex_model_backtest_sample_loss_dict
                self.complex_model_backtest_sample_loss_dict[model_name] = sample_loss

        # 计算简单融合模型的权重
        # 构造 simple_model_sample_weight_dict
        # 权重的计算方法是：样本损失越小，权重越大
        # 样本损失最小的模型的权重为决定系数 determine_ratio
        # 其他模型的权重按照样本损失的大小来分配

        

        









