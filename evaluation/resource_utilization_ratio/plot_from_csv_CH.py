import os

from py_plotter.plot import Plotter
import pandas as pd


my_plotter = Plotter(
    figsize=(10,8),
    font_thirdparty="YaHei",
    fontsize=30,
)
# my_plotter.color_list = ["#5EA0C7","#B4D7E5", "#6BC179", "#BEDEAB"]
my_plotter.color_list = ["#074166","#8CC5BE", "#CC011F", "#FADADD"]



def plot_figures(csv_filepath="./result.csv"):
    # 读取csv
    df = pd.read_csv(filepath_or_buffer=csv_filepath)
    print(f"df.head(): {df.head()}")

    # Histogram,IceBreaker,KeepAlive,SmartPred
    # Huawei,0.31,0.28,0.45,0.21
    # Crane,0.24,0.22,0.3,0.19
    # Azure,0.11,0.18,0.57,0.09

    # legend_label_list
    legend_label_list = ["Histogram", "IceBreaker", "KeepAlive", "SmartPred"]
    # x_data
    x_data = ["Huawei", "Crane", "Azure"]

    bar_data_list = []

    for legend_label in legend_label_list:
        bar_data = []

        for x in x_data:
            # Dataset列等于x的行，legend_label列的值
            item = df[df["Dataset"] == x][legend_label].values[0]
            bar_data.append(item)
        bar_data_list.append(bar_data)

    # 创建 save_root
    os.makedirs(f"./plot", exist_ok=True)

    my_plotter.plot_bars(
        x_data=x_data,
        bar_data_list=bar_data_list,
        legend_label_list=legend_label_list,
        y_min=0,
        y_max=1.19,
        x_label="数据集",
        y_label="资源利用率",
        filename="resource_utilization_ratio_CH.pdf",
        save_root="./plot",
        legend_title=None,
        legend_ncol=2,
    )





if __name__ == "__main__":
    plot_figures()