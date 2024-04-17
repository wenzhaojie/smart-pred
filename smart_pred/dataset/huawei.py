import os
import pandas as pd

class HuaweiPrivateDataset:
    def __init__(self, dataset_root:str="~/GitHubProjects/smart-pred/datasets/huawei/private_dataset"):
        # 绝对路径
        dataset_root = os.path.expanduser(dataset_root)
        print(f"dataset_root:{dataset_root}")
        self.dataset_root = dataset_root
        pass

    def load_and_cache_dataset(self, ):
        pass

    def get_all_function_name(self, ):
        # 获取所有函数名
        # day_000.csv 中的列名一共有
        csv_path = os.path.join(self.dataset_root, "cpu_limit_minute", "day_001.csv")
        df = pd.read_csv(csv_path)
        all_function_name = df.columns[2:]
        all_function_name_list = [str(x) for x in all_function_name]
        return all_function_name_list

    def get_num_of_function(self, ):
        # 获取函数数量
        num_of_function = len(self.get_all_function_name())
        return num_of_function


    def get_data_by_day(self, data_type: str, function_name: str, day: int, resolution: str):
        # private 的 day 000 到 234，确保day在这个范围内
        if day < 0 or day > 234:
            raise ValueError("day must be in range 0 to 234")
        # 确保resolution为"minute"或"second"
        if resolution not in ["minute", "second"]:
            raise ValueError("Resolution must be 'minute' or 'second'.")

        # 确保data_type有效
        valid_data_types = ["cpu_limit", "cpu_usage", "memory_limit", "memory_usage", "requests", "instances", "platform_delay", "function_delay"]
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type. Must be one of {valid_data_types}")

        # 构建文件路径
        folder_name = f"{data_type}_{resolution}"
        day_3_digit = f"{day:03d}"
        csv_file_name = f"day_{day_3_digit}.csv"
        csv_file_path = os.path.join(self.dataset_root, folder_name, csv_file_name)

        # 加载CSV
        df = pd.read_csv(csv_file_path)

        # 从列中筛选出需要的列
        df = df[[function_name, "day", "time"]]

        # 重命名列
        df = df.rename(columns={function_name: data_type})

        # 如果是分钟分辨率，将时间转换为分钟
        if resolution == "minute":
            df["time"] = df["time"] / 60

        # 返回list
        return df[data_type].tolist()

    def get_data_by_day_range(self, start_day, end_day, data_type: str, function_name: str, resolution: str):
        # 确保resolution为"minute"或"second"
        if resolution not in ["minute", "second"]:
            raise ValueError("Resolution must be 'minute' or 'second'.")

        # 确保data_type有效
        valid_data_types = ["cpu_limit", "cpu_usage", "memory_limit", "memory_usage", "requests", "instances", "platform_delay", "function_delay"]
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type. Must be one of {valid_data_types}")

        # 使用 get_data_by_day() 获取每一天的数据
        data_list = []
        for day in range(start_day, end_day + 1):
            data_list.extend(self.get_data_by_day(data_type, function_name, day, resolution))

        return data_list



class TestHuaweiPrivateDataset:

    def test_get_data_by_day(self):
        print("test_get_resource_data_by_day")
        dataset = HuaweiPrivateDataset()

        function_name = "150"
        day = 1
        resolution = "minute"

        data_type = "cpu_limit"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day(data_type, function_name, day, resolution)
        print(result)

        data_type = "requests"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day(data_type, function_name, day, resolution)
        print(result)

        data_type = "platform_delay"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day(data_type, function_name, day, resolution)
        print(result)

        data_type = "function_delay"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day(data_type, function_name, day, resolution)
        print(result)

        data_type = "memory_limit"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day(data_type, function_name, day, resolution)
        print(result)

        data_type = "instances"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day(data_type, function_name, day, resolution)
        print(result)

    def test_get_data_by_day_range(self):
        print("test_get_resource_data_by_day_range")
        dataset = HuaweiPrivateDataset()

        function_name = "150"
        start_day = 1
        end_day = 2
        resolution = "minute"

        data_type = "cpu_limit"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day_range(start_day, end_day, data_type, function_name, resolution)
        print(result)
        print(f"len(result): {len(result)}")

        data_type = "requests"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day_range(start_day, end_day, data_type, function_name, resolution)
        print(result)
        print(f"len(result): {len(result)}")


        data_type = "platform_delay"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day_range(start_day, end_day, data_type, function_name, resolution)
        print(result)
        print(f"len(result): {len(result)}")

        data_type = "function_delay"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day_range(start_day, end_day, data_type, function_name, resolution)
        print(result)
        print(f"len(result): {len(result)}")

        data_type = "memory_limit"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day_range(start_day, end_day, data_type, function_name, resolution)
        print(result)
        print(f"len(result): {len(result)}")

        data_type = "instances"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day_range(start_day, end_day, data_type, function_name, resolution)
        print(result)
        print(f"len(result): {len(result)}")


class HuaweiPublicDataset:
    # HuaweiPublicDataset只有requests_minute一个数据

    def __init__(self, dataset_root:str="~/GitHubProjects/smart-pred/datasets/huawei/public_dataset"):
        # 绝对路径
        dataset_root = os.path.expanduser(dataset_root)
        self.dataset_root = dataset_root
        pass

    def load_and_cache_dataset(self, ):
        pass

    def get_all_function_name(self, ):
        # 获取所有函数名
        # day_000.csv 中的列名一共有
        csv_path = os.path.join(self.dataset_root, "requests_minute", "day_01.csv")
        df = pd.read_csv(csv_path)
        all_function_name = df.columns[2:]
        all_function_name_list = [str(x) for x in all_function_name]
        return all_function_name_list

    def get_num_of_function(self, ):
        # 获取函数数量
        num_of_function = len(self.get_all_function_name())
        return num_of_function

    def get_data_by_day(self, data_type: str, function_name: str, day: int, resolution: str):
        # public 的 day 00 到 25，确保day在这个范围内
        if day < 0 or day > 25:
            raise ValueError("Day must be in range 0 to 25")

        # 确保resolution为"minute"
        if resolution not in ["minute",]:
            raise ValueError("Resolution must be 'minute'")

        # 确保data_type有效
        valid_data_types = ["requests"]
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type. Must be one of {valid_data_types}")

        # 构建文件路径
        folder_name = f"{data_type}_{resolution}"
        day_2_digit = f"{day:02d}"
        csv_file_name = f"day_{day_2_digit}.csv"
        csv_file_path = os.path.join(self.dataset_root, folder_name, csv_file_name)

        # 加载CSV
        df = pd.read_csv(csv_file_path)

        # 从列中筛选出需要的列
        df = df[[function_name, "day", "time"]]

        # 重命名列
        df = df.rename(columns={function_name: data_type})

        # 如果是分钟分辨率，将时间转换为分钟
        if resolution == "minute":
            df["time"] = df["time"] / 60

        # 返回list
        return df[data_type].tolist()

    def get_data_by_day_range(self, start_day, end_day, data_type: str, function_name: str, resolution: str):
        # 确保resolution为"minute"
        if resolution not in ["minute",]:
            raise ValueError("Resolution must be 'minute'")

        # 确保data_type有效
        valid_data_types = ["requests"]
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data_type. Must be one of {valid_data_types}")

        # 使用 get_data_by_day() 获取每一天的数据
        data_list = []
        for day in range(start_day, end_day + 1):
            data_list.extend(self.get_data_by_day(data_type, function_name, day, resolution))

        return data_list


class TestHuaweiPublicDataset:
    def __init__(self):
        pass

    def test_get_data_by_day(self):
        print("test_get_resource_data_by_day")
        dataset = HuaweiPublicDataset()

        function_name = "3"
        day = 1
        resolution = "minute"

        data_type = "requests"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day(data_type, function_name, day, resolution)
        print(result)

    def test_get_data_by_day_range(self):
        print("test_get_resource_data_by_day_range")
        dataset = HuaweiPublicDataset()

        function_name = "3"
        start_day = 1
        end_day = 2
        resolution = "minute"

        data_type = "requests"
        print(f"data_type: {data_type}")

        result = dataset.get_data_by_day_range(start_day, end_day, data_type, function_name, resolution)
        print(result)
        print(f"len(result): {len(result)}")


if __name__ == "__main__":
    # test = TestHuaweiPrivateDataset()
    # # test.test_get_data_by_day()
    # test.test_get_data_by_day_range()

    test = TestHuaweiPublicDataset()
    # test.test_get_data_by_day()
    test.test_get_data_by_day_range()
    pass


