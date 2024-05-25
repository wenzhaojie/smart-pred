# 这个文件是用来训练一个模型，它负责将不同预测算法的结果进行组合，得到最终的预测结果
import numpy as np

from smart_pred.solution.combo_model import ComboModel

import torch
from torch.utils.data import Dataset, DataLoader


# 现在开始训练这个模型
# 先生成数据集
# 数据集，假设有N个预测模型，他们都通过输入历史数据（长度为history_len），输出预测数据（长度为pred_len）。
# 此外，这N个预测模型在训练数据集上回测的准确率分别为Backtesting_loss1和Backtesting_loss2,...,Backtesting_lossN。
# 现在主要就是有以下几个模型："Avgvalue", "Maxvalue", "Movingavg", "Movingmax", "Crane_dsp_model",

# 我现在要生成数据集。首先，我们统一设置history_len=1440*4, pred_len=1440，即历史数据为4天，预测数据为1天
# 然后，我们在求Backtesting_loss的时候，我们用history_len=1440*3，pred_len=1440，即用前三天的历史数据作为训练，第四天的数据作为真实值，来回测检验模型的准确率
# Backtesting_loss 的值即为模型在第四天的MAE值

# 以下是使用我定义的一些库来获取负载的数据
from smart_pred.dataset.crane_trace import CraneDataset
from smart_pred.dataset.huawei import HuaweiPublicDataset, HuaweiPrivateDataset

trace_dict = {
        "period": [
            ("huawei_private", "4"),
            ("huawei_private", "5"),
            ("huawei_private", "15"),
            ("huawei_private", "25"),
            ("huawei_private", "33"),
            ("huawei_private", "39"),
            ("huawei_private", "40"),
            ("huawei_private", "60"),
            ("huawei_private", "72"),
            ("huawei_private", "75"),
            ("huawei_private", "92"),
            ("huawei_private", "100"),
            ("huawei_private", "116"),
            ("huawei_private", "129"),
        ],
        "continuous": [
            ("crane", "9"),
            ("crane", "2"),
            ("crane", "14"),
        ],
        "sparse": [
            ("crane", "10"),
            ("huawei_public", "43"),
            ("huawei_public", "51"),
            ("huawei_public", "61"),
            ("huawei_public", "97"),
        ],
        "bursty": [  # HUAWEI private 10,14,54,55,56,57,68,69,71
            ("huawei_private", "10"),
            ("huawei_private", "14"),
            ("huawei_private", "56"),
            ("huawei_private", "57"),
            ("huawei_private", "68"),
            ("huawei_private", "69"),
            ("huawei_private", "71"),
        ]
    }

def get_all_trace_from_dataset(start_day, end_day):
    all_trace_list = [] # 用来存储所有的trace, 其中每个trace是一个np.array，history+pred
    # 先遍历trace_dict获取所有的trace
    for pattern in ["period", "continuous", "sparse", "bursty"]:
        for dataset_name, trace_name in trace_dict[pattern]:
            if dataset_name == "huawei_private":
                dataset = HuaweiPrivateDataset()
            elif dataset_name == "huawei_public":
                dataset = HuaweiPublicDataset()
            elif dataset_name == "crane":
                dataset = CraneDataset()
            else:
                raise Exception(f"Unknown dataset name: {dataset_name}")

            trace = dataset.get_data_by_day_range(
                start_day=start_day,
                end_day=end_day,
                data_type="requests",
                function_name=trace_name,
                resolution="minute"
            )
            # 转换成np.array
            trace = np.array(trace)
            all_trace_list.append(trace)
    return all_trace_list

def generate_custom_dataset_with_csv(model_name_list, history_len=1440*4, pred_len=1440):
    # 用于准备好所有训练的数据，用csv来保存
    # 对于每一个method，需要准备的训练数据有两个
    # 第一个是使用前三天的数据作为输入，第四天的数据作为输出，用来训练模型，然后使用模型在第四天的数据上进行回测，得到模型的准确率（A）
    # 第二个是使用前四天的数据作为输入，第五天的数据作为输出得到的预测结果（B）。
    # X作为模型输入，Y作为模型输出
    # X包含所有不同模型的预测结果(B)，以及模型回测准确率(A)
    # Y包含第五天的真实值。

    # model_name_list = ["Avgvalue", "Maxvalue", "Movingavg", "Movingmax", "Dsp"]



    pass


class CustomDataset(Dataset):
    def __init__(self, all_trace_data):
        self.all_trace_data = all_trace_data

    def __len__(self):
        return len(self.all_trace_data)

    def __getitem__(self, index):
        # 关键在于这个
        return self.all_trace_data[index]

