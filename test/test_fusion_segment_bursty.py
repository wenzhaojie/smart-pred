from smart_pred.solution.fusion_algorithm.fusion_model_segment_bursty import Fusion_segment_bursty_model
from smart_pred.dataset.crane_trace import CraneDataset
from py_plotter.plot import Plotter


my_plotter = Plotter(
    figsize=(10, 6),
    font_thirdparty="YaHei",
)


class TestFusionSegmentBurstyModel:
    def __init__(self):

        self.fusion_model_params_dict = {
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
        self.model = Fusion_segment_bursty_model(fusion_model_params_dict=self.fusion_model_params_dict)


    def get_crane_trace_data(self, function_name="9"):
        # 获取数据
        crane_dataset = CraneDataset()
        data = crane_dataset.get_data_by_day_range(
            start_day=0,
            end_day=4,
            data_type="requests",
            function_name=function_name,
            resolution="minute"
        )
        print(f"len(data):{len(data)}")
        print(f"type(data):{type(data)}")
        return data


    def test_use_future_evaluation(self):
        # 用 crane 数据集中第9个函数来测试
        function_name = "9"
        data = self.get_crane_trace_data(function_name=function_name)

        print(f"len(data):{len(data)}")

        # 划分训练集和测试集，前4天为训练集，第5天为测试集
        train_data = data[0:1440*4]
        test_data = data[1440*4:1440*5]

        # extra_parameters
        extra_parameters = {
            "seq_len": 1440 * 4,
            "pred_len": 1440,
            "period_length": 1440,
            "complex_model_list": ["NHITS", "NBEATS", "PatchTST", "DLinear"],
            # "simple_model_list": ["Movingavg_model", "Movingmax_model", "Movingmin_model", "Maxvalue_model", "Minvalue_model", "Avgvalue_model", "Quantile_model"],
            "simple_model_list": ["Movingmax_model", "Movingavg_model", "Maxvalue_model", "Histogram_model"],
            "loss": "SelectiveAsymmetricMAELoss",
            "determine_ratio": 0.5,
            "is_scaler": False,
            "is_round": False,
            "amplify": 1.00,
            "is_complex": False,
            "delta_t": 120,
            "epsilon": 1.5,
            "bursty_quantile": 0.999,
        }

        # 训练
        self.model.train(history=train_data, extra_parameters=extra_parameters)

        # 预测
        log_dict, predict  = self.model.use_future_rolling_evaluation(train=train_data, test=test_data, extra_parameters=extra_parameters)
        print(f"log_dict:{log_dict}")
        print(f"predict:{predict}")
        print(f"true:{test_data}")

        plot_true = test_data[0:1440]
        plot_pred = predict[0:1440]

        # 画图
        x = [i for i in range(len(plot_true))]
        filename = "fusion_model_bursty_evaluation.png"
        my_plotter.plot_lines(
            x_list=[x, x],
            line_data_list=[plot_true, plot_pred],
            legend_label_list=["true", "pred"],
            filename=filename,
            x_label="time",
            y_label="value",
            save_root="plot",
            x_tick_ndigits=0,
            y_tick_ndigits=0,
        )

    def test_use_future_evaluation_2(self):
        # 用 crane 数据集中第9个函数来测试
        function_name = "9"
        data = self.get_crane_trace_data(function_name=function_name)

        print(f"len(data):{len(data)}")

        # 划分训练集和测试集，前4天为训练集，第5天为测试集
        train_data = data[0:1440*4]
        test_data = data[1440*4:1440*5]

        # extra_parameters
        extra_parameters = {
            "seq_len": 1440 * 4,
            "pred_len": 1440,
            "period_length": 1440,
            "complex_model_list": ["NHITS", "NBEATS", "PatchTST", "DLinear"],
            # "simple_model_list": ["Movingavg_model", "Movingmax_model", "Movingmin_model", "Maxvalue_model", "Minvalue_model", "Avgvalue_model", "Quantile_model"],
            "simple_model_list": ["Movingmax_model", "Movingavg_model", "Maxvalue_model", "Histogram_model"],
            "loss": "SelectiveAsymmetricMAELoss",
            "determine_ratio": 0.5,
            "is_scaler": False,
            "is_round": False,
            "amplify": 1.00,
            "is_complex": True,
            "delta_t": 120,
            "epsilon": 1.5,
            "bursty_quantile": 0.99,
        }

        # 训练
        self.model.train(history=train_data, extra_parameters=extra_parameters)

        # 预测
        log_dict, predict  = self.model.use_future_rolling_evaluation(train=train_data, test=test_data, extra_parameters=extra_parameters)
        print(f"log_dict:{log_dict}")
        print(f"predict:{predict}")
        print(f"true:{test_data}")

        plot_true = test_data[0:1440]
        plot_pred = predict[0:1440]

        # 画图
        x = [i for i in range(len(plot_true))]
        filename = "fusion_model_evaluation_is_complex.png"
        my_plotter.plot_lines(
            x_list=[x, x],
            line_data_list=[plot_true, plot_pred],
            legend_label_list=["true", "pred"],
            filename=filename,
            x_label="time",
            y_label="value",
            save_root="plot",
            x_tick_ndigits=0,
            y_tick_ndigits=0,
        )



if __name__ == "__main__":
    test_fusion_model = TestFusionSegmentBurstyModel()
    test_fusion_model.test_use_future_evaluation()
    # test_fusion_model.test_use_future_evaluation_2()
