import os

from py_plotter.plot import Plotter
import pandas as pd


my_plotter = Plotter(
    figsize=(10,8),
    font_thirdparty="YaHei",
    fontsize=30,
)
# my_plotter.color_list = ["#5EA0C7","#B4D7E5", "#6BC179", "#BEDEAB"]
# my_plotter.color_list = ["#9281DD", "#B7617D", "#E4B112", "#B7DB29", "#159FD7"]

# my_plotter.color_list = ["#55B7E6","#193E8F", "#E53528", "#F09739"]

my_plotter.color_list = ["#66BC98", "#AAD09D", "#E3EA96", "#FCDC89", "#E26844", "#8A233F"]





def plot_bar(dataset_name_list=None, method_name_list=None):

    # legend 是不同 method
    # x 轴是不同数据集
    bar_data_list = []
    # 读取数据
    for method in method_name_list:
        bar_data = []
        for dataset_name in dataset_name_list:
            csv_filepath = f"avg_{dataset_name}.csv"
            df = pd.read_csv(csv_filepath)
            # 读取数据
            train_t = df[df['method'] == method]['train_t'].values[0]
            predict_t = df[df['method'] == method]['predict_t'].values[0]
            # 如果 "Icebreaker", "Crane"，则乘以720
            if method in ["Icebreaker", "Crane"]:
                train_t *= 120
                predict_t *= 120
            total_t = train_t + predict_t
            bar_data.append(total_t)
        bar_data_list.append(bar_data)

    # 创建save_root
    if not os.path.exists("plot"):
        os.makedirs("plot")

    # x轴首字母大写
    dataset_name_list = [dataset_name.capitalize() for dataset_name in dataset_name_list]

    # 将用于画图的数据导出为csv
    df = pd.DataFrame(bar_data_list, columns=dataset_name_list)
    df['method'] = method_name_list
    # 只保留两位小数
    df = df.round(2)
    df.to_csv("plot/overhead_CH.csv", index=False)


    my_plotter.plot_bars(
        x_label="数据集",
        y_label="计算时间 (秒)",
        x_data=dataset_name_list,
        bar_data_list=bar_data_list,
        legend_label_list=method_name_list,
        filename="overhead_CH.pdf",
        save_root="plot",
        legend_title=None,
        legend_ncol=2,
        y_min=0,
        y_max=6,

    )


if __name__ == "__main__":
    plot_bar(
        dataset_name_list=["azure", "crane", "huawei"],
        method_name_list=['SmartPred', 'NHITS', 'NBEATS', "DLinear", "Icebreaker", "Crane"]
    )



