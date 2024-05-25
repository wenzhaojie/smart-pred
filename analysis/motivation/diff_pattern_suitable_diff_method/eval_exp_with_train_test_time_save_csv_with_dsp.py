# 这个实验用于说明一个观点，不存在一种特别牛的方法，对于所有类型的负载都有很好的效果
import json
import os

from smart_pred.dataset.crane_trace import CraneDataset
from smart_pred.dataset.huawei import HuaweiPublicDataset, HuaweiPrivateDataset

# method
from smart_pred.model.local.period import Maxvalue_model, Avgvalue_model
from smart_pred.model.local.passive import Movingavg_model, Movingmax_model
from smart_pred.model.local.neuralforecast_model import NeuralForecast_model
from smart_pred.model.local.fourier import Crane_dsp_model

from smart_pred.utils.metrics import get_metric_dict

import numpy as np
from copy import deepcopy

from py_plotter.plot import Plotter

my_plotter = Plotter(
    figsize=(20, 8),
    fontsize=30,
    font_thirdparty="YaHei",
)


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


def exp(start_day=0, end_day=4):
    # 所有的model
    # model_name_list = ["MLP", "NHITS", "NBEATS", "PatchTST", "TimesNet", "Maxvalue", "Avgvalue", "Movingavg", "Movingmax"]
    model_name_list = ["Dsp",]
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
            test = trace[1440*(end_day-start_day):1440*(end_day-start_day+1)]

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
                    start_train_t = time.time()
                    model.train(history=train, extra_parameters=extra_parameters)
                    end_train_t = time.time()
                    train_t = end_train_t - start_train_t
                    print(f"完成训练！")
                    # 预测
                    start_pred_t = time.time()
                    log_dict, predict = model.use_future_rolling_evaluation(train=train, test=test, extra_parameters=extra_parameters)
                    end_pred_t = time.time()
                    pred_t = end_pred_t - start_pred_t
                    # 保留2为小数
                    train_t = "{:.2f}".format(train_t)
                    pred_t = "{:.2f}".format(pred_t)

                    # 计算标准化之后的MAE
                    _pred = deepcopy(predict)
                    _test = deepcopy(test)

                    # 重新fit
                    model.scaler.fit_transform(train.reshape(-1, 1)).reshape(-1)
                    _pred = model.scaler.transform(_pred.reshape(-1, 1)).reshape(-1)
                    _test = model.scaler.transform(_test.reshape(-1, 1)).reshape(-1)
                    # get_metric_dict
                    mae = get_metric_dict(_test, _pred)["mae"] # 这里面已经把mae转换成了标准化之后的mae
                    # 保留两位小数
                    mae = "{:.2f}".format(mae)



                    # 绘制图像
                    x = np.arange(len(test))
                    pred = predict
                    true = test

                    x_list = [x, x]

                    seq_len = extra_parameters["seq_len"]
                    pred_len = extra_parameters["pred_len"]

                    # 生成文件名
                    file_name = f"{model_name}_{dataset_name}_{trace_name}_{pattern}_seq_len_{seq_len}_pred_len_{pred_len}_mae_{mae}_train_t_{train_t}_pred_t_{pred_t}.pdf"

                    # 如果y的最大值小于5，那么y_tick_ndigits=1
                    y_tick_ndigits = 0
                    if max(true) < 5:
                        y_tick_ndigits = 1

                    # 如果y大于 10000，那么使用科学计数法
                    is_y_tick_sci = False
                    if max(true) > 10000:
                        is_y_tick_sci = True

                    # 将pred和true 保存到csv文件中
                    # 创建 save_root
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    csv_filename = os.path.join(save_root, f"{file_name}.csv")
                    import pandas as pd
                    df = pd.DataFrame({
                        "pred": pred,
                        "true": true
                    })
                    df.to_csv(csv_filename, index=False)
                    print(f"已经保存 {csv_filename}!")

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
                        y_tick_ndigits=y_tick_ndigits,
                        is_y_tick_sci=is_y_tick_sci,
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


if __name__ == "__main__":
    exp()

