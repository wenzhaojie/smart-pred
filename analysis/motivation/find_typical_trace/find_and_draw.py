# 用于找到不同类型的典型trace，画出它们的样子

from smart_pred.dataset.huawei import HuaweiPrivateDataset
from smart_pred.dataset.huawei import HuaweiPublicDataset

from py_plotter.plot import Plotter

# 从私有数据集中获取数据
private_dataset = HuaweiPrivateDataset()
public_dataset = HuaweiPublicDataset()
my_plotter = Plotter(
    figsize=(30,6),
    fontsize=30,
)


def get_and_plot_private_function_trace(dataset, start_day, end_day, data_type, function_name, resolution, start_index=0, end_index=1000):
    """
    获取指定函数的追踪数据并绘制图形。

    :param dataset: 数据集对象，用于获取数据
    :param start_day: 开始天数
    :param end_day: 结束天数
    :param data_type: 数据类型（如 "requests"）
    :param function_name: 函数名称
    :param resolution: 分辨率（如 "minute"）
    :param plotter: 用于绘图的对象
    """
    # 获取追踪数据
    trace = dataset.get_data_by_day_range(start_day=start_day, end_day=end_day, data_type=data_type,
                                          function_name=function_name, resolution=resolution)

    # 处理NaN值，将其转换为0
    for i in range(len(trace)):
        if trace[i] != trace[i]:  # 判断是否为NaN
            trace[i] = 0

    # 截取部分数据
    if start_index is not None and end_index is not None:
        trace = trace[start_index:end_index]
        x = list(range(start_index, end_index))
    else:
        x = list(range(len(trace)))
    # 准备绘图数据

    file_name = f"private_{resolution}_{data_type}_{function_name}_day_{start_day}_{end_day}_start_index_{start_index}_end_index_{end_index}.pdf"

    # 使用绘图对象绘制线图
    my_plotter.plot_lines(
        x_list=[x],
        line_data_list=[trace],
        title=f"{data_type} of {function_name}",
        x_label="Time",
        y_label=data_type.capitalize(),
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")


def get_and_plot_public_function_trace(start_day, end_day, data_type, function_name, resolution, start_index=None, end_index=None):
    """
    获取指定函数的追踪数据并绘制图形。

    :param start_day: 开始天数
    :param end_day: 结束天数
    :param data_type: 数据类型（如 "requests"）
    :param function_name: 函数名称
    :param resolution: 分辨率（如 "minute"）
    :param plotter: 用于绘图的对象
    """
    # 获取追踪数据
    trace = public_dataset.get_data_by_day_range(start_day=start_day, end_day=end_day, data_type=data_type,
                                          function_name=function_name, resolution=resolution)

    # 处理NaN值，将其转换为0
    for i in range(len(trace)):
        if trace[i] != trace[i]:  # 判断是否为NaN
            trace[i] = 0

    # 截取部分数据
    if start_index is not None and end_index is not None:
        trace = trace[start_index:end_index]

    # 准备绘图数据
    x = list(range(len(trace)))
    file_name = f"public_{resolution}_{data_type}_{function_name}_day_{start_day}_{end_day}_start_index_{start_index}_end_index_{end_index}.pdf"

    # 使用绘图对象绘制线图
    my_plotter.plot_lines(
        x_list=[x],
        line_data_list=[trace],
        title=f"{data_type} of {function_name}",
        x_label="Time",
        y_label=data_type.capitalize(),
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")


def draw_bursty_trace():
    pass


def draw_periodic_trace(function_name="15", start_index=0, end_index=240):
    # private request 15 day 0 - 3
    get_and_plot_private_function_trace(
        dataset=private_dataset,
        start_day=0,
        end_day=0,
        data_type="requests",
        function_name=function_name,
        resolution="minute",
        start_index=start_index,
        end_index=end_index,
    )
    get_and_plot_private_function_trace(
        dataset=private_dataset,
        start_day=1,
        end_day=1,
        data_type="requests",
        function_name=function_name,
        resolution="minute",
        start_index=start_index,
        end_index=end_index,
    )
    get_and_plot_private_function_trace(
        dataset=private_dataset,
        start_day=2,
        end_day=2,
        data_type="requests",
        function_name=function_name,
        resolution="minute",
        start_index=start_index,
        end_index=end_index,
    )
    get_and_plot_private_function_trace(
        dataset=private_dataset,
        start_day=3,
        end_day=3,
        data_type="requests",
        function_name=function_name,
        resolution="minute",
        start_index=start_index,
        end_index=end_index,
    )
    pass



def draw_random_trace():
    pass


def draw_sparse_trace():
    pass



if __name__ == "__main__":
    draw_periodic_trace(function_name="4")
    draw_periodic_trace(function_name="15")
    draw_periodic_trace(function_name="16")
    draw_periodic_trace(function_name="25")
    draw_periodic_trace(function_name="28", start_index=0, end_index=1440)
    draw_periodic_trace(function_name="34", start_index=0, end_index=1440)
    draw_periodic_trace(function_name="39", start_index=0, end_index=1440)
    draw_periodic_trace(function_name="40", start_index=0, end_index=1440)
    draw_periodic_trace(function_name="60", start_index=0, end_index=1440)
    draw_periodic_trace(function_name="60", start_index=0, end_index=360)
    draw_periodic_trace(function_name="60", start_index=0, end_index=360)
    draw_periodic_trace(function_name="60", start_index=360, end_index=720)

    # function name 60 这个函数trace非常适合说明 bursty负载夹杂在一个连续序列上的情况，因此使用简单的滑动窗口即可实现较为准确的预测。但是不能准确捕获周期性负载。

    draw_periodic_trace(function_name="68", start_index=0, end_index=360)
    draw_periodic_trace(function_name="68", start_index=0, end_index=1440)

    pass