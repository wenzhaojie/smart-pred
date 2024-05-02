# 这个实验用于说明一个观点，不存在一种特别牛的方法，对于所有类型的负载都有很好的效果
import json
import os

from smart_pred.dataset.crane_trace import CraneDataset
from smart_pred.dataset.huawei import HuaweiPublicDataset, HuaweiPrivateDataset

# method
from smart_pred.model.local.period import Maxvalue_model, Avgvalue_model
from smart_pred.model.local.passive import Movingavg_model, Movingmax_model
from smart_pred.model.local.neuralforecast_model import NeuralForecast_model

import numpy as np

from py_plotter.plot import Plotter

my_plotter = Plotter(
    figsize=(10, 8),
    fontsize=30,
)


extra_parameter_dict = {
    "MLP": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": True,
        "is_round": False,
    },
    "NHITS": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": True,
        "is_round": False,
    },
    "NBEATS": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": True,
        "is_round": False,
    },
    "PatchTST": {
        "seq_len": 1440, # 1440*2 OOM
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": True,
        "is_round": False,
    },
    "TimesNet": {
        "seq_len": 1440*2,
        "pred_len": 1440,
        "max_steps": 100,
        "is_scaler": True,
        "is_round": False,
    },
    "Maxvalue": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": True,
        "is_round": False,
    },
    "Avgvalue": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": True,
        "is_round": False,
    },
    "Movingavg": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 10,
        "is_scaler": True,
        "is_round": False,
    },
    "Movingmax": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 10,
        "is_scaler": True,
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
}

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
    "bursty": [ # HUAWEI private 10,14,54,55,56,57,68,69,71
        ("huawei_private", "10"),
        ("huawei_private", "14"),
        ("huawei_private", "56"),
        ("huawei_private", "57"),
        ("huawei_private", "68"),
        ("huawei_private", "69"),
        ("huawei_private", "71"),
    ]
}


def exp(start_day=0, end_day=8):
    # 所有的model
    model_name_list = ["MLP", "NHITS", "NBEATS", "PatchTST", "TimesNet", "Maxvalue", "Avgvalue", "Movingavg", "Movingmax"]
    # model_name_list = ["TimesNet", "Maxvalue", "Movingavg",]
    # 所有的pattern
    pattern_list = ["period", "continuous", "sparse", "bursty"]
    # 存储结果
    result_dict_list = []

    # 生成一个
    # 当前时间 str
    import time
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_root = f"./plot_trace/{time_str}"

    # 遍历所有的pattern
    for pattern in pattern_list:
        print(f"pattern: {pattern}")
        for trace in trace_dict[pattern]:
            dataset_name, trace_name = trace
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

            # 划分训练集和测试集
            train = trace[0:1440*(end_day-start_day)]
            test = trace[1440*(end_day-start_day):1440*(end_day-start_day)+1]

            # 打印一下len
            print(f"len(train): {len(train)}")
            print(f"len(test): {len(test)}")

            for model_name in model_name_list:
                model = model_dict[model_name]
                extra_parameters = extra_parameter_dict[model_name]

                print(f"model_name: {model_name}")
                print(f"trace_name: {trace_name}")
                print(f"extra_parameters: {extra_parameters}")

                try:
                    # 训练模型
                    model.train(history=train, extra_parameters=extra_parameters)
                    print(f"完成训练！")
                    # 预测
                    log_dict, predict = model.use_future_rolling_evaluation(train=train, test=test, extra_parameters=extra_parameters)
                    # MAE
                    mae = log_dict["mae"]
                    # 保留2位小数
                    mae = "{:.2f}".format(mae)

                    # 绘制图像
                    x = np.arange(len(test))
                    pred = predict
                    true = test


                    # 将画图的数据保存在csv中
                    # 创建文件夹
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    csv_filename = os.path.join(save_root, f"{model_name}_{dataset_name}_{trace_name}_{pattern}_plot_data.csv")
                    import pandas as pd
                    df = pd.DataFrame({
                        "x": x,
                        "pred": pred,
                        "true": true,
                    })
                    df.to_csv(csv_filename, index=False)

                    seq_len = extra_parameters["seq_len"]
                    pred_len = extra_parameters["pred_len"]

                    # 生成文件名
                    file_name = f"{model_name}_{dataset_name}_{trace_name}_{pattern}_seq_len_{seq_len}_pred_len_{pred_len}_mae_{mae}.pdf"
                    # 将extra_parameters写入json
                    json_file_name = os.path.join(save_root, f"{model_name}_{dataset_name}_{trace_name}_{pattern}_seq_len_{seq_len}_pred_len_{pred_len}_extra_parameters.json")
                    import json
                    with open(json_file_name, "w") as f:
                        json.dump(extra_parameters, f)
                    print(f"已经保存 {json_file_name}!")

                    x_list = [x, x]
                    my_plotter.plot_lines(
                        x_list=x_list,
                        line_data_list=[pred, true],
                        legend_label_list=["Predict", "True"],
                        legend_title=None,
                        title=None,
                        x_grid=True,
                        y_grid=True,
                        x_label="Time",
                        y_label="Requests",
                        save_root=save_root,
                        filename=file_name,
                        x_tick_ndigits=0,
                        y_tick_ndigits=0,
                    )
                    print(f"已经绘制 {file_name}!")

                    # 记录结果
                    result_dict = {
                        "model_name": model_name,
                        "trace_name": trace_name,
                        "pattern": pattern,
                        "log_dict": log_dict,
                        "extra_parameters": extra_parameters,
                    }
                    print(result_dict)
                except Exception as e:
                    raise e

                # 保存结果
                result_dict_list.append(result_dict)

                # 保存csv
                import pandas as pd
                df = pd.DataFrame(result_dict_list)

                csv_filename = os.path.join(save_root, "result.csv")
                df.to_csv(csv_filename, index=False)
                print(f"已经保存 {csv_filename}!")


def draw_prediction(save_root="./plot_trace", json_file_name=None):
    # json_file_name = os.path.join(save_root, f"{model_name}_{dataset_name}_{trace_name}_{pattern}_seq_len_{seq_len}_pred_len_{pred_len}_extra_parameters.json")
    with open(json_file_name, "r") as f:
        extra_parameters = json.load(f)
    print(f"extra_parameters: {extra_parameters}")
    # load csv
    import pandas as pd
    # csv_filename = os.path.join(save_root, f"{model_name}_{dataset_name}_{trace_name}_{pattern}_plot_data.csv")

    # 提取文件名部分
    file_name = os.path.basename(json_file_name)

    # 去除文件扩展名
    file_name_without_extension = os.path.splitext(file_name)[0]

    # 使用 "_" 分割文件名
    parts = file_name_without_extension.split("_")

    # 提取各个参数
    model_name = parts[0]
    dataset_name = parts[1]
    trace_name = parts[2]
    pattern = parts[3]
    seq_len_index = parts.index("seq") + 1
    seq_len = int(parts[seq_len_index])
    pred_len_index = parts.index("pred") + 1
    pred_len = int(parts[pred_len_index])

    csv_filename = os.path.join(save_root, f"{model_name}_{dataset_name}_{trace_name}_{pattern}_seq_len_{seq_len}_pred_len_{pred_len}_plot_data.csv")

    df = pd.read_csv(csv_filename)
    x = df["x"].values
    pred = df["pred"].values
    true = df["true"].values

    x_list = [x, x]
    file_name = f"{model_name}_{dataset_name}_{trace_name}_{pattern}_seq_len_{seq_len}_pred_len_{pred_len}_mae_{mae}.pdf"

    my_plotter.plot_lines(
        x_list=x_list,
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=None,
        x_grid=True,
        y_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root=save_root,
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")



if __name__ == "__main__":
    exp()

