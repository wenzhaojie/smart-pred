from smart_pred.solution.fusion_algorithm.fusion_model import Fusion_model
from smart_pred.dataset.crane_trace import CraneDataset


class TestFusionModel:
    def __init__(self):
        self.model = Fusion_model()


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

        # extra_parameters
        extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "period_length": 1440,
            "complex_model_list": ["NHITS", "NBEATS", "PatchTST", "DLinear"],
            "simple_model_list": ["Movingavg_model", "Movingmax_model", "Movingmin_model", "Maxvalue_model", "Minvalue_model", "Avgvalue_model", "Quantile_model"],
            "loss": "MSELoss",
            "is_scaler": False,
            "is_round": False,
        }

        # 训练
        self.model.train(history=data, extra_parameters=extra_parameters)






if __name__ == "__main__":
    test_fusion_model = TestFusionModel()
    test_fusion_model.test_train()