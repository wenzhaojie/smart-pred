from smart_pred.model.local.statsforecast_model import StatsForecast_model
from smart_pred.dataset.crane_trace import CraneDataset
from py_plotter.plot import Plotter


my_plotter = Plotter(
    figsize=(10, 6),
    font_thirdparty="YaHei",
)


class TestArimaModel:
    def __init__(self):
        self.model = StatsForecast_model(name="AutoARIMA")


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


    def test_train(self):
        # 用 crane 数据集中第9个函数来测试
        function_name = "9"
        data = self.get_crane_trace_data(function_name=function_name)

        # 前4天的数据用于训练，第5天的数据用于预测
        history = data[:-1440]
        true = data[-1440:]

        # extra_parameters
        extra_parameters = {
            "seq_len": 1440 * 2,
            "pred_len": 1440,
            "loss": "MSELoss",
            "is_scaler": False,
            "is_round": False,
            "season_length": 1440,
        }

        # 训练
        self.model.train(history=history, extra_parameters=extra_parameters)

        # 预测
        predict_window = 1440
        pred = self.model.predict(history=history, predict_window=predict_window, extra_parameters=extra_parameters)

        # 画图
        legend_label_list = ["pred", "true"]
        x_list = []
        for _ in legend_label_list:
            x = range(len(pred))
            x_list.append(x)

        line_data_list = []
        line_data_list.append(pred)
        line_data_list.append(true)

        my_plotter.plot_lines(
            x_list=x_list,
            legend_label_list=legend_label_list,
            line_data_list=line_data_list,
            filename="test_arima_model.png",
            save_root="./"
        )






if __name__ == "__main__":
    test_arima_model = TestArimaModel()
    test_arima_model.test_train()