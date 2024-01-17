# 处理azure trace 2021数据集
import os
import math
import time
import pickle
import multiprocessing

import numpy as np
import pandas as pd

class AzureFunction2021:
    def __init__(self, dataset_root: str, original_file_name: str):
        self.dataset_root = dataset_root
        self.original_file_name = original_file_name
        self.original_df = None
        self.processed_df = None
        self.app_df_dict = None

    def load_original_csv(self):
        csv_path = os.path.join(self.dataset_root, self.original_file_name)
        df = pd.read_csv(csv_path)
        self.original_df = df

    def process_csv(self):
        # 深拷贝一份df
        processed_df = self.original_df.copy()

        # 在原始df中添加一列，表示函数调用开始的时间，叫 start_timestamp
        processed_df["start_timestamp"] = processed_df["end_timestamp"] - processed_df["duration"]

        # 统计所有不重复的app
        app_set = set(processed_df["app"])
        app_list = list(app_set)

        # 为每一个app创建一个df，用 app_df_dict 存储
        app_df_dict = {}
        for app in app_list:
            app_df_dict[app] = processed_df[processed_df["app"] == app]

        # 将处理好的df存储到类中
        self.app_df_dict = app_df_dict

    def generate_invocation_time_series_by_sec(self, app_name: str, function_name: str):
        # 从app_df_dict中取出对应的df
        df = self.app_df_dict[app_name]

        # 从df中取出对应的function的df
        function_df = df[df["func"] == function_name]

        # 生成时间序列
        # start_index是向下取整，end_index是向上取整
        start_index = int(function_df["start_timestamp"].min())
        end_index = int(function_df["start_timestamp"].max()) + 1

        # 初始化时间序列
        invocation_time_series = [0 for _ in range(0, end_index - start_index + 1)]

        # 遍历function_df，每一行都是一个函数调用，在其start_timestamp的位置上，invocation_time_series中对应的时间段加1
        for index, row in function_df.iterrows():
            start = int(row["start_timestamp"])
            invocation_time_series[start - start_index] += 1

        # 返回时间序列
        return invocation_time_series

    # 定义函数以计算一个app中一个function的duration的分位数
    def cal_app_function_duration_percentile(self, app_name: str, function_name: str, percentile=99):
        # 从 app_df_dict 中取出对应的 df
        df = self.app_df_dict[app_name]

        # 筛选特定 function 的数据
        specific_func_df = df[df['func'] == function_name]

        duration_in_percentile = np.percentile(specific_func_df['duration'], percentile)

        # 计算并返回分位数
        return duration_in_percentile


    # 定义函数以计算一个app中所有function的duration的分位数
    def cal_app_all_function_duration_percentile(self, app_name: str, percentile=99):
        # 从 app_df_dict 中取出对应的 df
        df = self.app_df_dict[app_name]

        # 筛选特定 function 的数据
        duration_in_percentile = np.percentile(df['duration'], percentile)

        # 计算并返回分位数
        return duration_in_percentile

    # 定义函数以计算一个app中一个function，在某一个timestamp，同时有多少个function在运行
    def cal_app_function_concurrency_at_timestamp(self, app_name: str, function_name: str, timestamp: float):
        # 从 app_df_dict 中取出对应的 df
        df = self.app_df_dict[app_name]

        # 筛选特定 function 的数据
        specific_func_df = df[df['func'] == function_name]

        # 筛选特定 timestamp 的数据
        specific_timestamp_df = specific_func_df[specific_func_df['start_timestamp'] <= timestamp]
        specific_timestamp_df = specific_timestamp_df[specific_timestamp_df['end_timestamp'] >= timestamp]

        # 计算并返回并发数
        concurrency_at_timestamp = len(specific_timestamp_df)

        # 计算并返回并发数
        return concurrency_at_timestamp

    # 定义函数以计算一个app中一个function，在每一个秒内，平均有多少个function在同时运行，精确到毫秒
    def cal_app_function_avg_concurrency_in_sec_with_ms(self, app_name: str, function_name: str) -> list:
        # 从 app_df_dict 中取出对应的 df
        df = self.app_df_dict[app_name]

        # 筛选特定 function 的数据
        specific_func_df = df[df['func'] == function_name]

        # 换算一共有多少毫秒
        end_timestamp = math.ceil(df['end_timestamp'].max()) # 单位为秒
        num_of_milliseconds = (end_timestamp) * 1000

        # result_list 用于存储每一毫秒的并发数
        # 初始化 result_list
        result_list = [0 for _ in range(0, num_of_milliseconds)]
        # 遍历df中的每一行，每一行都是一个函数调用，将其start_timestamp和end_timestamp之间的时间段内的并发数加1
        for index, row in specific_func_df.iterrows():
            start = math.floor(row["start_timestamp"] * 1000)
            end = math.ceil(row["end_timestamp"] * 1000)
            for i in range(start, end):
                result_list[i] += 1

        # result_list转换为秒级的并发数，每秒为1000个毫秒中的最大并发数
        result_list = [max(result_list[i:i+1000]) for i in range(0, len(result_list), 1000)]

        # 计算并返回平均并发数
        return result_list




class TestAzureFunction2021:
    def __init__(self):
        self.dataset_root = r"C:\Users\wenzh\PycharmProjects\smart-pred\datasets\AzureFunctionsInvocationTraceForTwoWeeksJan2021"
        self.original_file_name = "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
        self.azure_function_2021 = AzureFunction2021(self.dataset_root, self.original_file_name)

    def test_load_original_csv(self):
        print(f"开始测试 {self.azure_function_2021.original_file_name} 的 load_original_csv 函数")
        self.azure_function_2021.load_original_csv()
        print(self.azure_function_2021.original_df.head())


    def test_process_csv(self):
        print(f"开始测试 {self.azure_function_2021.original_file_name} 的 process_csv 函数")
        self.azure_function_2021.load_original_csv()
        self.azure_function_2021.process_csv()
        print(self.azure_function_2021.app_df_dict.keys())


    def test_generate_invocation_time_series_by_sec(self):
        print(f"开始测试 {self.azure_function_2021.original_file_name} 的 generate_invocation_time_series_by_sec 函数")
        self.azure_function_2021.load_original_csv()
        self.azure_function_2021.process_csv()
        app_name = "7b2c43a2bc30f6bb438074df88b603d2cb982d3e7961de05270735055950a568"
        function_name = "e3cdb48830f66eb8689cc0223514569a69812b77e6611e3d59814fac0747bd2f"
        self.azure_function_2021.generate_invocation_time_series_by_sec(app_name, function_name)
        time_series = self.azure_function_2021.generate_invocation_time_series_by_sec(app_name, function_name)


        # 一共调用了多少次
        length = len(time_series)
        num_of_invocations = sum(time_series)
        print(f"一共调用了 {num_of_invocations} 次")
        print(f"一共有 {length} 个时间段")


    def test_cal_app_function_duration_percentile(self):
        print(f"开始测试 {self.azure_function_2021.original_file_name} 的 cal_app_function_duration_percentile 函数")
        self.azure_function_2021.load_original_csv()
        self.azure_function_2021.process_csv()
        app_name = "7b2c43a2bc30f6bb438074df88b603d2cb982d3e7961de05270735055950a568"
        function_name = "e3cdb48830f66eb8689cc0223514569a69812b77e6611e3d59814fac0747bd2f"
        percentile = 99
        duration_in_percentile = self.azure_function_2021.cal_app_function_duration_percentile(app_name, function_name, percentile)
        print(f"{app_name} 中 {function_name} 的 duration 的 {percentile} 分位数是 {duration_in_percentile}")

    def test_cal_app_all_function_duration_percentile(self):
        print(f"开始测试 {self.azure_function_2021.original_file_name} 的 cal_app_all_function_duration_percentile 函数")
        self.azure_function_2021.load_original_csv()
        self.azure_function_2021.process_csv()
        app_name = "7b2c43a2bc30f6bb438074df88b603d2cb982d3e7961de05270735055950a568"
        percentile = 99
        duration_in_percentile = self.azure_function_2021.cal_app_all_function_duration_percentile(app_name, percentile)
        print(f"{app_name} 中所有 function 的 duration 的 {percentile} 分位数是 {duration_in_percentile}")

    def test_cal_app_function_concurrency_at_timestamp(self):
        print(f"开始测试 {self.azure_function_2021.original_file_name} 的 cal_app_function_concurrency_at_timestamp 函数")
        self.azure_function_2021.load_original_csv()
        self.azure_function_2021.process_csv()
        app_name = "7b2c43a2bc30f6bb438074df88b603d2cb982d3e7961de05270735055950a568"
        function_name = "e3cdb48830f66eb8689cc0223514569a69812b77e6611e3d59814fac0747bd2f"
        timestamp = 0.05
        start_t = time.time()
        concurrency_at_timestamp = self.azure_function_2021.cal_app_function_concurrency_at_timestamp(app_name, function_name, timestamp)
        print(f"{app_name} 中 {function_name} 在 {timestamp} 时刻的并发数是 {concurrency_at_timestamp}")
        end_t = time.time()
        print(f"耗时 {end_t - start_t} s")

    def test_cal_app_function_avg_concurrency_in_sec_with_ms(self):
        print(f"开始测试 {self.azure_function_2021.original_file_name} 的 cal_app_function_avg_concurrency_in_sec_with_ms 函数")
        self.azure_function_2021.load_original_csv()
        self.azure_function_2021.process_csv()
        # app_name = "7b2c43a2bc30f6bb438074df88b603d2cb982d3e7961de05270735055950a568"
        # function_name = "e3cdb48830f66eb8689cc0223514569a69812b77e6611e3d59814fac0747bd2f"
        # start_t = time.time()
        # avg_concurrency_list = self.azure_function_2021.cal_app_function_avg_concurrency_in_sec_with_ms(app_name, function_name)
        # end_t = time.time()
        # print(f"耗时 {end_t - start_t} s")
        # total_invocations = sum(avg_concurrency_list)
        # print(f"{app_name} 中 {function_name} 一共调用了 {total_invocations} 次")

        # 下一个测试
        app_name = "70b9cea7ca266637479483f517194c402dfe99b5fc2357e6ebac5e715c9a34a2"
        function_name = "30aa434528bc68ee07745ee7be3a0bdb33d58961fdc8460ce5b5b46b4def96e8"
        start_t = time.time()
        avg_concurrency_list = self.azure_function_2021.cal_app_function_avg_concurrency_in_sec_with_ms(app_name, function_name)
        end_t = time.time()
        print(f"耗时 {end_t - start_t} s")
        total_invocations = sum(avg_concurrency_list)
        print(f"{app_name} 中 {function_name} 一共调用了 {total_invocations} 次")


def process_app(args):
    dataset, app_name, pickle_root = args
    # app对应的所有function_name
    function_name_list = list(dataset.app_df_dict[app_name]["func"].unique())
    result_dict = {}

    for function_name in function_name_list:
        avg_concurrency_in_sec_with_ms_list = dataset.cal_app_function_avg_concurrency_in_sec_with_ms(app_name=app_name, function_name=function_name)
        result_dict[function_name] = avg_concurrency_in_sec_with_ms_list

    # pickle_path = app_name + ".pickle"
    pickle_path = os.path.join(pickle_root, f"{app_name}.pickle")

    # 将result_dict 保存 pickle
    with open(pickle_path, "wb") as f:
        pickle.dump(result_dict, f)

    print(f"{app_name} 处理完成")





def generate_processed_dataset(num_of_app=10, output_pickle_root="../analysis/processed_dataset"):
    # 创建output_csv_root文件夹
    if not os.path.exists(output_pickle_root):
        os.mkdir(output_pickle_root)


    # 利用多进程来处理数据集
    # 我们按照每一个app来划分数据集，每一个app的处理任务作为一个进程任务

    dataset_obj = AzureFunction2021(
        dataset_root=r"../datasets/AzureFunctionsInvocationTraceForTwoWeeksJan2021",
        original_file_name="AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    )
    dataset_obj.load_original_csv()
    dataset_obj.process_csv()

    # app的set list
    app_set_list = list(dataset_obj.app_df_dict.keys())
    print(f"一共有 {len(app_set_list)} 个app")

    # 取前num_of_app个app, 如果大于app的总数，就取所有的app
    # 如果num_of_app为None，就取所有的app
    if num_of_app is not None:
        if num_of_app > len(app_set_list):
            num_of_app = len(app_set_list)
        app_set_list = app_set_list[:num_of_app]

    # 使用 multiprocessing.Pool 创建进程池
    # 获得cpu核心数
    cpu_count = multiprocessing.cpu_count()
    print(f"cpu核心数为 {cpu_count} 个")

    # 准备输入列表，其中每个元素都是一个元组 (dataset_obj, app_name, pickle_root)
    map_input_list = []
    for app_name in app_set_list:
        map_input_list.append((dataset_obj, app_name, output_pickle_root))

    with multiprocessing.Pool(processes=12) as pool:
        map_output_list = pool.map(process_app, map_input_list)

    print(map_output_list)



if __name__ == "__main__":
    test = TestAzureFunction2021()
    # test.test_load_original_csv()
    # test.test_process_csv()
    # test.test_generate_invocation_time_series_by_sec()
    # test.test_cal_app_function_duration_percentile()
    # test.test_cal_app_all_function_duration_percentile()
    # test.test_cal_app_function_concurrency_at_timestamp()
    # test.test_cal_app_function_avg_concurrency_in_sec_with_ms()

    generate_processed_dataset(num_of_app=120)
















