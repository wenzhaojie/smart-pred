import os
import re

import pandas as pd
from matplotlib import pyplot as plt
from py_plotter.plot import Plotter

my_plotter = Plotter(
    figsize=(10, 8),
    dpi=300,
    fontsize=30,
    font_thirdparty="YaHei",
)

def stat_data():
    # 文件名列表
    file_names = [
        "Avgvalue_crane_2_continuous_seq_len_4320_pred_len_1440_mae_0.13_train_t_0.00_pred_t_0.02.pdf.csv",
        "MLP_crane_2_continuous_seq_len_2880_pred_len_1440_mae_0.46_train_t_1.02_pred_t_0.13.pdf.csv",
        "Maxvalue_crane_2_continuous_seq_len_4320_pred_len_1440_mae_0.10_train_t_0.00_pred_t_0.01.pdf.csv",
        "Movingavg_crane_2_continuous_seq_len_4320_pred_len_1_mae_0.26_train_t_0.00_pred_t_0.02.pdf.csv",
        "Movingmax_crane_2_continuous_seq_len_4320_pred_len_1_mae_0.26_train_t_0.00_pred_t_0.02.pdf.csv",
        "NBEATS_crane_2_continuous_seq_len_2880_pred_len_1440_mae_0.71_train_t_2.35_pred_t_0.18.pdf.csv",
        "NHITS_crane_2_continuous_seq_len_2880_pred_len_1440_mae_0.57_train_t_2.05_pred_t_0.16.pdf.csv",
        "PatchTST_crane_2_continuous_seq_len_1440_pred_len_1440_mae_0.42_train_t_42.02_pred_t_0.31.pdf.csv",
        "TimesNet_crane_2_continuous_seq_len_2880_pred_len_1440_mae_0.45_train_t_530.24_pred_t_0.29.pdf.csv"
    ]

    # 提取train_t和pred_t的正则表达式模式
    pattern = r'mae_([\d.]+)_train_t_([\d.]+)_pred_t_([\d.]+)\.'

    # 存储模型名称及其对应的train_t和pred_t值
    model_stats = {}

    # 解析每个文件名
    for file_name in file_names:
        # 使用正则表达式匹配train_t和pred_t的值
        match = re.search(pattern, file_name)
        if match:
            mae = float(match.group(1))
            train_t = float(match.group(2))
            pred_t = float(match.group(3))

            # 提取模型名称
            model_name = file_name.split("_")[0]
            # 存储模型名称及其对应的train_t和pred_t值
            model_stats[model_name] = (mae, train_t, pred_t)

    # 写入csv
    # 最开始一行是表头
    with open("model_stats.csv", "a") as f:
        f.write("Model,MAE,Train_t,Pred_t\n")

        # 打印模型名称及其对应的train_t和pred_t值
        for model, stats in model_stats.items():
            print(f"Model: {model}, mae: {stats[0]}, train_t: {stats[1]}, pred_t: {stats[2]}")
            f.write(f"{model},{stats[0]},{stats[1]},{stats[2]}\n")
            pass
        pass


def plot_non_dl_data():
    # 读取CSV文件
    df = pd.read_csv("model_stats.csv")

    # 提取模型名称
    model_name_list = ["Avgvalue", "Maxvalue", "Movingavg", "Movingmax"]

    mae_value_list = []
    compute_t_value_list = []

    # 提取 MAE、Train_t 和 Pred_t 数据
    for model_name in model_name_list:
        model_data = df[df['Model'].str.startswith(model_name)]
        mae_value_list.append(model_data['MAE'].iloc[0])
        compute_t_value_list.append(model_data['Train_t'].iloc[0] + model_data['Pred_t'].iloc[0])

    # 绘制图表
    my_plotter = Plotter(
        figsize=(10, 8),
        dpi=300,
        fontsize=30,
        font_thirdparty="YaHei",
    )
    # 创建文件夹save_root
    if not os.path.exists("./results"):
        os.makedirs("./results")
    my_plotter.plot_bars_and_lines(
        x_data=model_name_list,
        bar_y_label="MAE",
        line_y_label="Train_t + Pred_t",
        bar_data_list=[mae_value_list],
        line_data_list=[compute_t_value_list],
        legend_label_list=None,
        save_root="./results",
        filename="non_dl_data_plot",
        legend_title="Legend",
        bar_y_min=min(mae_value_list),
        bar_y_max=max(mae_value_list),
        line_y_min=min(compute_t_value_list),
        line_y_max=max(compute_t_value_list),
    )


if __name__ == '__main__':
    # stat_data()
    plot_non_dl_data()
    pass