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
    file_names = os.listdir("./plot_trace")

    # 提取train_t和pred_t的正则表达式模式
    pattern = r'mae_([\d.]+)_train_t_([\d.]+)_pred_t_([\d.]+)\.'

    # 存储模型名称及其对应的train_t和pred_t值
    model_stats_list = []

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
            model_stats_list.append((model_name, mae, train_t, pred_t))

    # 写入csv
    # 最开始一行是表头
    with open("model_stats.csv", "a") as f:
        f.write("Model,MAE,Train_t,Pred_t\n")

        # 打印模型名称及其对应的train_t和pred_t值
        for model_name, mae, train_t, pred_t in model_stats_list:
            f.write(f"{model_name},{mae},{train_t},{pred_t}\n")


def plot_non_dl_data():
    # 读取CSV文件
    df = pd.read_csv("model_stats.csv")

    # 提取模型名称
    model_name_list = ["Avgvalue", "Maxvalue", "Movingavg", "Movingmax", "Dsp"]

    mae_value_list = []
    compute_t_value_list = []

    # 提取 MAE、Train_t 和 Pred_t 数据
    for model_name in model_name_list:
        model_data = df[df['Model'].str.startswith(model_name)]
        print(f"model_data: {model_data}")
        # 取平均值
        mae_value_list.append(model_data['MAE'].mean())
        compute_t_value_list.append((model_data['Train_t'] + model_data['Pred_t']).mean())

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

    # compute_t_value_list 乘 1000 变成 ms
    compute_t_value_list = [x * 1000 for x in compute_t_value_list]
    my_plotter.plot_bars_and_lines(
        x_label="Model",
        x_data=model_name_list,
        bar_y_label="MAE",
        line_y_label="Total Compute Time (ms)",
        bar_data_list=[mae_value_list],
        line_data_list=[compute_t_value_list],
        legend_label_list=None,
        save_root="./results",
        filename="non_dl_data_plot.pdf",
        legend_title="Legend",
        bar_y_min=0,
        bar_y_max=1.0,
        line_y_min=0,
        line_y_max=30,
        line_y_tick_ndigits=0,
        is_hatch=True,
        is_marker=True,
        marker_size=30,
        x_label_rotation=20,
    )

def plot_dl_data():
    # 读取CSV文件
    df = pd.read_csv("model_stats.csv")

    # 提取模型名称
    model_name_list = ["PatchTST", "NHITS", "NBEATS", "MLP"]

    mae_value_list = []
    compute_t_value_list = []

    # 提取 MAE、Train_t 和 Pred_t 数据
    for model_name in model_name_list:
        model_data = df[df['Model'].str.startswith(model_name)]
        # 取平均值
        mae_value_list.append(model_data['MAE'].mean())
        compute_t_value_list.append((model_data['Train_t'] + model_data['Pred_t']).mean())

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
        x_label="Model",
        x_data=model_name_list,
        bar_y_label="MAE",
        line_y_label="Total Compute Time (s)",
        bar_data_list=[mae_value_list],
        line_data_list=[compute_t_value_list],
        legend_label_list=None,
        save_root="./results",
        filename="dl_data_plot.pdf",
        legend_title="Legend",
        bar_y_min=0,
        bar_y_max=1.2,
        line_y_min=0,
        line_y_max=50,
        line_y_tick_ndigits=0,
        is_hatch=True,
        is_marker=True,
        marker_size=30,
        x_label_rotation=20,
    )



if __name__ == '__main__':
    stat_data()
    plot_non_dl_data()
    plot_dl_data()
    pass