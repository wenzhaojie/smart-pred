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


def generate_result_csv(dataset_name_list=None, baseline_list=None, metric="cold_start_invocation_ratio"):
    if dataset_name_list is None:
        dataset_name_list = []
    if baseline_list is None:
        baseline_list = []

    # 用于存储所有行的列表
    result_rows = []

    # 遍历所有数据集
    for dataset_name in dataset_name_list:
        # 构建每个文件的文件路径
        csv_filepath = f"avg_{dataset_name}.csv"

        # 读取CSV文件
        df = pd.read_csv(csv_filepath)

        # 检查baseline中的每个方法是否存在于df中
        for baseline in baseline_list:
            if baseline in df['method'].values:
                # 提取特定的指标
                metric_value = df.loc[df['method'] == baseline, metric].values[0]
                # 将结果添加到行列表中
                result_rows.append({
                    'method': baseline,
                    'dataset_name': dataset_name,
                    'metric': metric_value
                })

    # 创建一个DataFrame来保存所有结果
    result_df = pd.DataFrame(result_rows)

    # 定义结果CSV的文件名
    result_csv_filename = f"result_{metric}.csv"

    # 保存结果到CSV文件
    result_df.to_csv(result_csv_filename, index=False)
    print(f"Results saved to {result_csv_filename}")




def plot_cold_start_figure(csv_filepath="./result_cold_start_invocation_ratio.csv"):
    # 读取csv
    df = pd.read_csv(csv_filepath)

    # legend_label_list，确保和方法名一致
    legend_label_list = ["SmartPred", "Icebreaker", "Keepalive", "Crane"]
    # x_data，保证和数据集名称一致
    x_data = ["huawei", "crane", "azure"]

    # 初始化bar_data_list
    bar_data_list = [[] for _ in legend_label_list]

    # 遍历每个数据集
    for dataset in x_data:
        # 遍历每个方法
        for i, method in enumerate(legend_label_list):
            # 提取对应的metric值
            metric_value = df[(df['dataset_name'] == dataset) & (df['method'] == method)]['metric'].values
            # 换算成百分比
            metric_value = [x * 100 for x in metric_value]
            if len(metric_value) > 0:  # 确保有值
                bar_data_list[i].append(metric_value[0])
            else:
                bar_data_list[i].append(0)  # 如果没有值，则默认为0

    # 创建 save_root
    os.makedirs("./plot", exist_ok=True)

    # dataset名称首字母大写
    x_data = [x.capitalize() for x in x_data]

    # 假设你已经有一个名为my_plotter的对象用于绘制图形
    my_plotter.plot_bars(
        x_data=x_data,
        bar_data_list=bar_data_list,
        legend_label_list=legend_label_list,
        x_label="数据集",
        y_label="冷启动概率（%）",
        filename="cold_start_ratio_CH.pdf",
        save_root="./plot",
        legend_title=None,
        legend_ncol=1,
        y_tick_ndigits=0,
        y_min=0,
        y_max=100,
    )


def plot_utilization_figure(csv_filepath="./result_utilization_ratio.csv"):
    # 读取csv
    df = pd.read_csv(csv_filepath)

    # legend_label_list，确保和方法名一致
    legend_label_list = ["SmartPred", "Icebreaker", "Keepalive", "Crane"]
    # x_data，保证和数据集名称一致
    x_data = ["huawei", "crane", "azure"]

    # 初始化bar_data_list
    bar_data_list = [[] for _ in legend_label_list]

    # 遍历每个数据集
    for dataset in x_data:
        # 遍历每个方法
        for i, method in enumerate(legend_label_list):
            # 提取对应的metric值
            metric_value = df[(df['dataset_name'] == dataset) & (df['method'] == method)]['metric'].values
            # 换算成百分比
            metric_value = [x * 100 for x in metric_value]
            if len(metric_value) > 0:  # 确保有值
                bar_data_list[i].append(metric_value[0])
            else:
                bar_data_list[i].append(0)  # 如果没有值，则默认为0

    # 创建 save_root
    os.makedirs("./plot", exist_ok=True)

    # dataset名称首字母大写
    x_data = [x.capitalize() for x in x_data]

    # 假设你已经有一个名为my_plotter的对象用于绘制图形
    my_plotter.plot_bars(
        x_data=x_data,
        bar_data_list=bar_data_list,
        legend_label_list=legend_label_list,
        x_label="数据集",
        y_label="资源利用率（%）",
        filename="resource_utilization_ratio_CH.pdf",
        save_root="./plot",
        legend_title=None,
        legend_ncol=1,
        y_max=119,
        y_min=0,
        y_tick_ndigits=0,
    )




if __name__ == "__main__":
    generate_result_csv(
        dataset_name_list=["azure", "huawei", "crane"],
        baseline_list=["SmartPred", "Crane", "Keepalive", "Icebreaker"],
        metric="cold_start_invocation_ratio"
    )

    generate_result_csv(
        dataset_name_list=["azure", "huawei", "crane"],
        baseline_list=["SmartPred", "Crane", "Keepalive", "Icebreaker"],
        metric="utilization_ratio"
    )

    plot_cold_start_figure()
    plot_utilization_figure()