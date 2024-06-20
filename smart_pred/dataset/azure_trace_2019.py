from tqdm import tqdm
import os
import pandas as pd

class AzureFunction2019:
    def __init__(self, dataset_root: str="~/GitHubProjects/smart-pred/datasets/azurefunctions-dataset2019"):
        dataset_root = os.path.expanduser(dataset_root)
        self.dataset_root = dataset_root
        self.invocation_df_dict = {}
        self.duration_df_dict = {}
        self.memory_df_dict = {}
        pass

    def load_and_cache_dataset(self, ):
        print(f"正在加载数据集!")
        for day in tqdm(range(1, 14 + 1)):
            invocation_filename = f"invocations_per_function_md.anon.d{str(day).rjust(2, '0')}.csv"
            duration_filename = f"function_durations_percentiles.anon.d{str(day).rjust(2, '0')}.csv"
            memory_filename = f"app_memory_percentiles.anon.d{str(day).rjust(2, '0')}.csv"

            invocation_filepath = os.path.join(self.dataset_root, invocation_filename)
            duration_filepath = os.path.join(self.dataset_root, duration_filename)

            invocation_df = pd.read_csv(invocation_filepath)
            duration_df = pd.read_csv(duration_filepath)

            if day <= 12:
                memory_filepath = os.path.join(self.dataset_root, memory_filename)
                memory_df = pd.read_csv(memory_filepath)
            else:
                memory_df = None

            self.invocation_df_dict[str(day)] = invocation_df
            self.duration_df_dict[str(day)] = duration_df
            self.memory_df_dict[str(day)] = memory_df

        print(f"数据集加载完成!")

    def get_invocation_df_by_day(self, day: int):
        return self.invocation_df_dict[str(day)]

    def get_duration_df_by_day(self, day: int):
        return self.duration_df_dict[str(day)]

    def get_memory_df_by_day(self, day: int):
        return self.memory_df_dict[str(day)]


    def get_hash_function_list_by_day(self, day):
        # 找到某一天全部的hash_function, 返回值是 hash_function_list
        invocation_df = self.invocation_df_dict[str(day)]
        hash_function_list = invocation_df["HashFunction"].tolist()
        print(type(hash_function_list))
        return hash_function_list


    def get_hash_function_set_appear_in_all_days(self):
        # 找到在每一天都出现的 hash_function_set
        find_all_hash_function_set = set()
        for day in range(1, 14 + 1):
            if day == 1:
                find_all_hash_function_set = set(self.get_hash_function_list_by_day(day))
                continue
            per_day_hash_function_set = set(self.get_hash_function_list_by_day(day))
            find_all_hash_function_set = per_day_hash_function_set & find_all_hash_function_set

        return find_all_hash_function_set


    def get_all_invocation_trace_by_hash_function(self, hash_function):
        # 拼接某一个hash_function的14天全部trace, 以分钟为单位
        all_invocation_list = []
        for day in range(1, 14 + 1):
            invocation_df_day = self.invocation_df_dict[str(day)]
            df_slice = invocation_df_day.loc[invocation_df_day["HashFunction"] == hash_function]
            invocation_list = []
            # 遍历每一分钟
            for i in range(1, 1440 + 1):
                is_invocation = int(df_slice[str(i)].iloc[0])
                invocation_list.append(is_invocation)
            all_invocation_list.extend(invocation_list)
        return all_invocation_list



class TestAzureFunction2019:
    def __init__(self):
        pass

    def test_load_and_cache_dataset(self, ):
        print(f"test_load_and_cache_dataset!")
        dataset_root = "~/GitHubProjects/smart-pred/datasets/azurefunctions-dataset2019"
        azure_function = AzureFunction2019(dataset_root)
        azure_function.load_and_cache_dataset()
        pass

    def test_get_invocation_df_by_day(self, ):
        print(f"test_get_invocation_df_by_day!")
        dataset_root = "~/GitHubProjects/smart-pred/datasets/azurefunctions-dataset2019"
        azure_function = AzureFunction2019(dataset_root)
        azure_function.load_and_cache_dataset()
        invocation_df = azure_function.get_invocation_df_by_day(1)
        print(invocation_df)
        pass

    def test_get_hash_function_list_by_day(self, ):
        print(f"test_get_hash_function_list_by_day!")
        dataset_root = "~/GitHubProjects/smart-pred/datasets/azurefunctions-dataset2019"
        azure_function = AzureFunction2019(dataset_root)
        azure_function.load_and_cache_dataset()
        hash_function_list = azure_function.get_hash_function_list_by_day(1)
        print(hash_function_list)
        pass

    def test_get_hash_function_set_appear_in_all_days(self, ):
        print(f"test_get_hash_function_set_appear_in_all_days!")
        dataset_root = "~/GitHubProjects/smart-pred/datasets/azurefunctions-dataset2019"
        azure_function = AzureFunction2019(dataset_root)
        azure_function.load_and_cache_dataset()
        hash_function_set = azure_function.get_hash_function_set_appear_in_all_days()
        print(hash_function_set)
        pass

    def test_get_all_invocation_trace_by_hash_function(self, ):
        print(f"test_get_all_invocation_trace_by_hash_function!")
        dataset_root = "~/GitHubProjects/smart-pred/datasets/azurefunctions-dataset2019"
        azure_function = AzureFunction2019(dataset_root)
        azure_function.load_and_cache_dataset()
        hash_function_set = azure_function.get_hash_function_set_appear_in_all_days()
        hash_function = list(hash_function_set)[100]
        all_invocation_trace = azure_function.get_all_invocation_trace_by_hash_function(hash_function)
        print(all_invocation_trace)
        pass



if __name__ == '__main__':
    test = TestAzureFunction2019()
    # test.test_load_and_cache_dataset()
    # test.test_get_invocation_df_by_day()
    # test.test_get_hash_function_list_by_day()
    # test.test_get_hash_function_set_appear_in_all_days()
    test.test_get_all_invocation_trace_by_hash_function()
