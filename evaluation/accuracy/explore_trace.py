# 用来测试不同方法的accuracy
# 我们采用这样的方式，先测试不同的trace，比较不同的方法，然后确定模型的参数来调优。
# 最后，通过统计的方式来计算平均的accuracy，生成画图的数据。
from smart_pred.dataset.huawei import HuaweiPublicDataset, HuaweiPrivateDataset
from py_plotter.plot import Plotter


my_plotter = Plotter(
    figsize=(20,8),
    font_thirdparty="YaHei"
)

huawei_public_ds = HuaweiPublicDataset()
huawei_private_ds = HuaweiPrivateDataset()

def explore_public_trace(start_day=0, end_day=13):
    # Public 10,13,16,3,6
    selected_public_trace = [10,13,16,3,6]

    # 绘制图片
    for function_name in selected_public_trace:
        trace = huawei_public_ds.get_data_by_day_range(
            start_day=start_day,
            end_day=end_day,
            data_type="requests",
            resolution="minute",
            function_name=str(function_name)
        )
        # 处理NaN值，将其转换为0
        for i in range(len(trace)):
            if trace[i] != trace[i]:  # 判断是否为NaN
                trace[i] = 0

        # 准备绘图数据
        x = list(range(len(trace)))
        file_name = f"selected_public_trace_{function_name}_day_{start_day}_{end_day}.pdf"

        # 使用绘图对象绘制线图
        my_plotter.plot_lines(
            x_list=[x],
            line_data_list=[trace],
            x_label="Time",
            y_label="Requests",
            save_root="./public/plot_trace",
            x_tick_ndigits=0,
            y_tick_ndigits=0,
            filename=file_name
        )
    pass


def explore_private_trace(start_day=0, end_day=4):
    # Private 150，160，161，33，72
    selected_private_trace = [150, 160, 161, 33, 72] + [4,5,15,25,33,34,39,40,60,72,73,75,86] + [
            28,39,40,60,72,73,75,86,88,91,92,94,96,97,100,102,104,107,116,124,126,127,129,130,131,132,135,136,146,147,148,150,155,163,179,183,190,194,195,199
        ]

    # 绘制图片
    for function_name in selected_private_trace:
        trace = huawei_public_ds.get_data_by_day_range(
            start_day=start_day,
            end_day=end_day,
            data_type="requests",
            resolution="minute",
            function_name=str(function_name)
        )
        # 处理NaN值，将其转换为0
        for i in range(len(trace)):
            if trace[i] != trace[i]:  # 判断是否为NaN
                trace[i] = 0

        # 准备绘图数据
        x = list(range(len(trace)))
        file_name = f"selected_private_trace_{function_name}_day_{start_day}_{end_day}.pdf"

        # 使用绘图对象绘制线图
        my_plotter.plot_lines(
            x_list=[x],
            line_data_list=[trace],
            x_label="Time",
            y_label="Requests",
            save_root="./private/plot_trace",
            x_tick_ndigits=0,
            y_tick_ndigits=0,
            filename=file_name
        )
    pass

if __name__ == "__main__":
    explore_public_trace()
    explore_private_trace()