# 这个文件是用来训练一个模型，它负责将不同预测算法的结果进行组合，得到最终的预测结果
import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import optim

from smart_pred.solution.combo_model import ComboModel

import torch
from torch.utils.data import Dataset, DataLoader

# method
from smart_pred.model.local.period import Maxvalue_model, Avgvalue_model
from smart_pred.model.local.passive import Movingavg_model, Movingmax_model
from smart_pred.model.local.neuralforecast_model import NeuralForecast_model
from smart_pred.model.local.fourier import Crane_dsp_model

from smart_pred.utils.metrics import get_metric_dict
from sklearn.preprocessing import StandardScaler


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

extra_parameter_dict = {
    "MLP": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": False,
        "is_round": False,
    },
    "NHITS": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": False, # 打开scaler效果会很差
        "is_round": False,
    },
    "NBEATS": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": False, # 打开scaler效果会很差
        "is_round": False,
    },
    "PatchTST": {
        "seq_len": 1440, # 1440*2 OOM
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": False, # 打开scaler效果会很差
        "is_round": False,
    },
    "TimesNet": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": False, # 打开scaler效果会很差
        "is_round": False,
    },
    "Maxvalue": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    },
    "Avgvalue": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    },
    "Movingavg": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 10,
        "is_scaler": False,
        "is_round": False,
    },
    "Movingmax": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 10,
        "is_scaler": False,
        "is_round": False,
    },
    "Dsp": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    },
}

model_dict = {
    "MLP": NeuralForecast_model(name="MLP"),
    "NHITS": NeuralForecast_model(name="NHITS"),
    "NBEATS": NeuralForecast_model(name="NBEATS"),
    "PatchTST": NeuralForecast_model(name="PatchTST"),
    "TimesNet": NeuralForecast_model(name="TimesNet"),
    "Maxvalue": Maxvalue_model(),
    "Avgvalue": Avgvalue_model(),
    "Movingavg": Movingavg_model(),
    "Movingmax": Movingmax_model(),
    "Dsp": Crane_dsp_model(name="Dsp"),
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
            # 处理NaN
            for i in range(len(trace)):
                if trace[i] != trace[i]:
                    trace[i] = 0
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

    # 创建一个pd dataframe，用来存储某一个负载的数据。不同的负载序列，需要创建不同的dataframe

    # 创建一个 dataframe，有以下列：model_name, backtesting_loss, model_pred, true_pred
    # model_name: 模型的名字
    # backtesting_loss: 模型在第四天的回测损失
    # model_pred: 模型在第五天的预测结果
    # true_pred: 第五天的真实值
    all_trace_list = get_all_trace_from_dataset(0, 4)

    # print len(all_trace_list)
    print(f"len(all_trace_list): {len(all_trace_list)}")

    for index, trace in enumerate(all_trace_list):
        # 对于每一个trace，我们需要创建一个dataframe
        df = pd.DataFrame(columns=["model_name", "backtesting_loss", "model_pred", "true_pred"])
        for model_name in model_name_list:
            # 对于每一个模型，我们需要计算backtesting_loss
            history_for_backtesting = trace[:(history_len-1440)]
            true_for_backtesting = trace[(history_len-1440):history_len]

            extra_parameters = extra_parameter_dict[model_name]
            # 训练模型
            model = model_dict[model_name]
            model.train(
                history=history_for_backtesting,
                extra_parameters=extra_parameters
            )
            # 预测
            _, predict = model.use_future_rolling_evaluation(
                train=history_for_backtesting,
                test=true_for_backtesting,
                extra_parameters=extra_parameters
            )
            # 计算backtesting_loss
            backtesting_loss = get_metric_dict(true_for_backtesting, predict)["mae"]

            # 模型的预测结果为pred
            # 现在进行第二个训练
            history_for_pred = trace[:history_len]
            true_for_pred = trace[history_len:]
            # 训练模型
            model.train(
                history=history_for_pred,
                extra_parameters=extra_parameters
            )
            # 预测
            _, predict = model.use_future_rolling_evaluation(
                train=history_for_pred,
                test=true_for_pred,
                extra_parameters=extra_parameters
            )
            # model_pred = 模型在第五天的预测结果
            model_pred = predict
            # 我们需要将这些数据加入到df中
            # 检查一下这些数据的长度
            assert len(model_pred) == 1440
            assert len(true_for_pred) == 1440
            assert backtesting_loss >= 0
            assert model_name in model_name_list

            # 将这些数据加入到df中, 对于model_pred和true_for_pred，我们需要将其转换成字符串
            model_pred_str = json.dumps(model_pred.tolist())
            true_for_pred_str = json.dumps(true_for_pred.tolist())
            print(f"model_pred_str: {model_pred_str}")
            print(f"true_for_pred_str: {true_for_pred_str}")

            one_row_dict = {
                "model_name": model_name,
                "backtesting_loss": backtesting_loss,
                "model_pred": model_pred_str,
                "true_pred": true_for_pred_str
            }
            new_row_df = pd.DataFrame([one_row_dict])
            # 将one_row_dict添加到DataFrame中
            df = pd.concat([df, new_row_df], ignore_index=True)

        # 现在我们需要将这个df保存到csv中
        save_root = './trace_dataset'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        csv_file_path = os.path.join(save_root, f"trace_index_{index}.csv")
        df.to_csv(csv_file_path)
        pass
    pass


class CustomDataset(Dataset):
    def __init__(self, csv_file_dir, model_name_list):
        self.csv_file_dir = csv_file_dir
        self.model_name_list = model_name_list
        self.num_files = len(os.listdir(csv_file_dir))

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        # 根据index获取csv文件
        csv_file_path = os.path.join(self.csv_file_dir, f"trace_index_{index}.csv")
        # 读取csv文件
        df = pd.read_csv(csv_file_path)
        # 获取model_name, backtesting_loss, model_pred, true_pred
        model_name_dict_list = []
        for model_name in self.model_name_list:
            # 获取model_name下的backtesting_loss, model_pred, true_pred
            model_name_series = df[df["model_name"] == model_name]
            backtesting_loss = model_name_series["backtesting_loss"].values[0]
            model_pred = json.loads(model_name_series["model_pred"].values[0])
            true_pred = json.loads(model_name_series["true_pred"].values[0])

            # 归一化 model_pred 和 true_pred
            scaler = StandardScaler()
            true_pred = scaler.fit_transform(np.array(true_pred).reshape(-1, 1)).reshape(-1)
            model_pred = scaler.transform(np.array(model_pred).reshape(-1, 1)).reshape(-1)
            # 还原list
            true_pred = true_pred.tolist()
            model_pred = model_pred.tolist()

            model_name_dict = {
                "model_name": model_name,
                "backtesting_loss": backtesting_loss,
                "model_pred": model_pred,
                "true_pred": true_pred
            }
            model_name_dict_list.append(model_name_dict)

        # 返回不同model_name下, backtesting_loss, model_pred, true_pred
        return model_name_dict_list


class Exp:
    def __init__(self, model_name_list):
        self.custom_dataset = None
        self.model_name_list = model_name_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.combo_model = ComboModel(num_models=len(model_name_list), pred_len=1440).to(self.device)
        self.optimizer = optim.Adam(self.combo_model.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def init_dataloader(self):
        # Construct CustomDataset
        csv_file_dir = "./trace_dataset"
        self.custom_dataset = CustomDataset(
            csv_file_dir=csv_file_dir,
            model_name_list=self.model_name_list
        )

    def train(self, epochs):
        self.combo_model.train()
        dataloader = DataLoader(self.custom_dataset, batch_size=10, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, data in enumerate(dataloader):
                self.optimizer.zero_grad()
                # 构造模型的输入
                nn_model_input = []
                true_pred = None
                for model_data in data:
                    model_pred = torch.tensor(model_data["model_pred"]).float().to(self.device)
                    true_pred = torch.tensor(model_data["true_pred"]).float().to(self.device)
                    backtesting_loss = torch.tensor(model_data["backtesting_loss"]).float().to(self.device)

                    # nn_model_input 的维度为 (num_models, pred_len+1)
                    nn_model_input.append(torch.cat((model_pred, backtesting_loss), dim=0))
                nn_model_input = torch.stack(nn_model_input)
                true_pred = true_pred.unsqueeze(0)


                # Forward pass
                pred_combo = self.combo_model.forward(
                    x=nn_model_input
                )

                # Compute loss
                loss = self.criterion(pred_combo, true_pred)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")

    def plot_samples(self, dataloader, num_samples=5):
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i >= num_samples:
                    break
                # 构造模型的输入
                nn_model_input = []
                true_pred = None
                for model_data in data:
                    model_pred = torch.tensor(model_data["model_pred"]).float().to(self.device)
                    true_pred = torch.tensor(model_data["true_pred"]).float().to(self.device)
                    backtesting_loss = torch.tensor(model_data["backtesting_loss"]).float().to(self.device)

                    # nn_model_input 的维度为 (num_models, pred_len+1)
                    nn_model_input.append(torch.cat((model_pred, backtesting_loss), dim=0))
                nn_model_input = torch.stack(nn_model_input)
                true_pred = true_pred.unsqueeze(0)

                # Forward pass
                pred_combo = self.combo_model.forward(
                    x=nn_model_input
                )

                # 绘制预测值和真实值的对比图
                plt.figure()
                plt.plot(pred_combo.squeeze().cpu().numpy(), label='Prediction')
                plt.plot(true_pred.squeeze().cpu().numpy(), label='True')
                plt.title(f'Sample {i + 1} Prediction vs True')
                plt.legend()
                plt.show()


if __name__ == "__main__":
    # # 先生成数据集
    # generate_custom_dataset_with_csv(
    #     model_name_list = ["Avgvalue", "Maxvalue", "Movingavg", "Movingmax", "Dsp"],
    #     history_len=1440*4,
    #     pred_len=1440
    # )

    # 然后训练模型
    exp = Exp(model_name_list=["Avgvalue", "Maxvalue", "Movingavg", "Movingmax", "Dsp"])
    exp.init_dataloader()
    exp.train(epochs=100)
    # 然后绘制一些样本
    exp.plot_samples(dataloader=DataLoader(exp.custom_dataset, batch_size=1, shuffle=True), num_samples=5)
    pass
