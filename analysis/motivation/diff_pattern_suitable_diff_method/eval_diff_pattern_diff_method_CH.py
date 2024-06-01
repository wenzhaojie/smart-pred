# 统计不同pattern的负载序列，不同模型的性能
# 已经有的实验结果位于plot_trace里
import os
import csv
from py_plotter.plot import Plotter


my_plotter = Plotter(
    figsize=(20,8),
    dpi=300,
    font_thirdparty="YaHei",
)
# 八中不同的bar颜色
my_plotter.color_list = color_list = ['#43978F', '#9EC4BE', '#ABD0F1', '#DCE9F4', '#E56F5E', '#F19685', '#F6C957', '#FFB77F', '#FBE8D5']

my_plotter.edge_color_list = ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']


# 找到所有的csv文件
# csv命名规则 Avgvalue_crane_2_continuous_seq_len_4320_pred_len_1440_mae_0.13_train_t_0.00_pred_t_0.02.pdf.csv

pattern_label_dict = {
    "continuous": "连续型",
    "sparse": "稀疏型",
    "period": "周期型",
    "bursty": "突增型",
}



def get_pattern_method_avg_mae(pattern, method, plot_trace_dir):
    method_mae_list = []

    # 找到plot_trace_dir目录下所有的csv文件
    csv_files = os.listdir(plot_trace_dir)

    for csv_file in csv_files:
        if method in csv_file:
            if pattern in csv_file:
                # 从csv文件名中解析出mae
                print(csv_file)
                mae = csv_file.split("mae")[1].split("_")[1]
                print(mae)
                method_mae_list.append(float(mae))
    # 计算平均mae
    avg_mae = sum(method_mae_list) / len(method_mae_list)
    res_dict = {
        "pattern": pattern,
        "method": method,
        "avg_mae": avg_mae,
    }
    return res_dict


def stat_all_data():
    res_dict_list = []
    for method in ["Avgvalue", "Maxvalue", "Dsp", "NBEATS", "Movingavg", "Movingmax", "NHITS", "PatchTST", "TimesNet"]:
        for pattern in ["continuous", "sparse", "period", "bursty"]:
            res_dict = get_pattern_method_avg_mae(pattern, method, "./plot_trace/2024-05-25-16-29-18")
            print(res_dict)
            res_dict_list.append(res_dict)

    # 保存至csv，使用csv模块
    csv_file_name = "pattern_method_avg_mae.csv"
    with open(csv_file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        writer.writerow(["pattern", "method", "avg_mae"])
        # 写入多行用writerows
        writer.writerows([[res_dict["pattern"], res_dict["method"], res_dict["avg_mae"]] for res_dict in res_dict_list])

    print("save to csv file: ", csv_file_name)

# 画图
def plot_pattern_method_avg_mae():
    # x轴是method，y轴是avg_mae，不同的pattern用不同的颜色
    # 读取csv文件
    csv_file_name = "pattern_method_avg_mae.csv"
    # 使用pandas读取csv文件
    import pandas as pd
    df = pd.read_csv(csv_file_name)
    print(df)
    # 画图
    bar_data_list = []
    legend_label_list = []
    for pattern in df["pattern"].unique():
        _pattern = pattern_label_dict[pattern]
        legend_label_list.append(_pattern)
    for pattern in df["pattern"].unique():
        bar_data = df[df["pattern"] == pattern]["avg_mae"].tolist()
        bar_data_list.append(bar_data)

    x_data = []
    for method in df["method"].unique():
        x_data.append(method)
    save_root = "./results"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    my_plotter.plot_bars(
        x_label="方法",
        y_label="平均 MAE",
        x_data=x_data,
        bar_data_list=bar_data_list,
        legend_label_list=legend_label_list,
        legend_title=None,
        save_root=save_root,
        filename="plot_pattern_method_avg_mae_CH.pdf",
        x_label_rotation=45,
        legend_ncol=2,
    )

    pass


def plot_method_pattern_avg_mae():
    # x轴是pattern，y轴是avg_mae，不同的method用不同的颜色
    # 读取csv文件
    csv_file_name = "pattern_method_avg_mae.csv"
    # 使用pandas读取csv文件
    import pandas as pd
    df = pd.read_csv(csv_file_name)
    print(df)
    # 画图
    bar_data_list = []
    # legend_label_list = df["method"].unique()
    # legend_label_list = ['Avgvalue', 'Maxvalue', 'Movingavg', 'Movingmax', 'NBEATS', 'NHITS', 'PatchTST', 'TimesNet']
    legend_label_list = ['Movingavg', 'Movingmax', 'NBEATS', 'NHITS', 'PatchTST', 'TimesNet']
    print(legend_label_list)
    for method in legend_label_list:
        bar_data = df[df["method"] == method]["avg_mae"].tolist()
        bar_data_list.append(bar_data)

    x_data = []
    for pattern in df["pattern"].unique():
        pattern = pattern_label_dict[pattern]
        x_data.append(pattern)
    save_root = "./results"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    my_plotter.plot_bars(
        x_label="负载特征",
        y_label="平均 MAE",
        x_data=x_data,
        bar_data_list=bar_data_list,
        legend_label_list=legend_label_list,
        legend_title=None,
        save_root=save_root,
        filename="plot_method_pattern_avg_mae_CH.pdf",
        legend_ncol=1,
    )


if __name__ == '__main__':
    # get_pattern_method_avg_mae("continuous","NBEATS", "./plot_trace/2024-05-19-20-48-21")
    # stat_all_data()
    plot_pattern_method_avg_mae()
    plot_method_pattern_avg_mae()

