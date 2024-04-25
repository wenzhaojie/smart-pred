import os
import pandas as pd

class CraneDataset:
    def __init__(self, dataset_root:str="~/GitHubProjects/smart-pred/datasets/crane"):
        # 绝对路径
        dataset_root = os.path.expanduser(dataset_root)
        print(f"dataset_root:{dataset_root}")
        self.dataset_root = dataset_root
        pass

    def load_and_cache_dataset(self, ):
        pass

    def get_all_function_name(self, ):
        # 获取所有函数名
        all_function_time = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        return all_function_time

    def get_num_of_function(self, ):
        # 获取函数数量
        num_of_function = len(self.get_all_function_name())
        return num_of_function

    def get_data_by_day(self, data_type: str, function_name: str, day: int, resolution: str):
        # crane 数据集一共 16天的数据，day的范围是0-15
        if day < 0 or day > 15:
            raise ValueError("day must be in range 0 to 15")
        # 确保resolution为"minute"或"second"
        if resolution not in ["minute"]:
            raise ValueError("Resolution must be 'minute'.")

        # 确保data_type有效
        valid_data_types = ["requests",]
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type. Must be one of {valid_data_types}")

        # 构建文件路径
        csv_file_name = f"input{function_name}.csv"
        csv_file_path = os.path.join(self.dataset_root, csv_file_name)

        # 加载CSV
        df = pd.read_csv(csv_file_path)

        # 从列中筛选出需要的列，一共两列，ts和value
        value_data = df["value"]

        # 筛选出day的数据
        value_data = value_data[day * 1440 : (day + 1) * 1440]

        # 返回list
        return value_data.tolist()

    def get_data_by_day_range(self, start_day, end_day, data_type: str, function_name: str, resolution: str):
        # 确保resolution为"minute"或"second"
        if resolution not in ["minute",]:
            raise ValueError("Resolution must be 'minute'.")

        # 确保data_type有效
        valid_data_types = ["requests",]
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type. Must be one of {valid_data_types}")

        # 使用 get_data_by_day() 获取每一天的数据
        data_list = []
        for day in range(start_day, end_day + 1):
            data_list.extend(self.get_data_by_day(data_type, function_name, day, resolution))

        return data_list