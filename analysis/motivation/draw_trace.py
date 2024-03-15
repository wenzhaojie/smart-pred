# 从直观的来看看华为的trace都长啥样

from smart_pred.dataset.huawei import HuaweiPrivateDataset
from smart_pred.dataset.huawei import HuaweiPublicDataset

from py_plotter.plot import Plotter

# 从私有数据集中获取数据
private_dataset = HuaweiPrivateDataset()
public_dataset = HuaweiPublicDataset()
my_plotter = Plotter()


def get_and_plot_private_function_trace(dataset, start_day, end_day, data_type, function_name, resolution):
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

    # 准备绘图数据
    x = list(range(len(trace)))
    file_name = f"private_{resolution}_{data_type}_{function_name}_day_{start_day}_{end_day}.pdf"

    # 使用绘图对象绘制线图
    my_plotter.plot_lines(
        x_list=[x],
        line_data_list=[trace],
        title=f"{data_type} of {function_name}",
        x_label="Time",
        y_label=data_type.capitalize(),
        save_root="./plot_trace",
        filename=file_name
    )
    print(f"已经绘制 {file_name}!")


def get_and_plot_public_function_trace(start_day, end_day, data_type, function_name, resolution):
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

    # 准备绘图数据
    x = list(range(len(trace)))
    file_name = f"public_{resolution}_{data_type}_{function_name}_day_{start_day}_{end_day}.pdf"

    # 使用绘图对象绘制线图
    my_plotter.plot_lines(
        x_list=[x],
        line_data_list=[trace],
        title=f"{data_type} of {function_name}",
        x_label="Time",
        y_label=data_type.capitalize(),
        save_root="./plot_trace",
        filename=file_name
    )
    print(f"已经绘制 {file_name}!")



def plot_all_trace_private():
    start_day = 0
    end_day = 18
    data_type = "requests"
    resolution = "minute"

    for function_name in private_dataset.get_all_function_name():
        get_and_plot_private_function_trace(private_dataset, start_day, end_day, data_type, function_name, resolution)


def plot_all_trace_public():
    start_day = 0
    end_day = 21
    data_type = "requests"
    resolution = "minute"

    for function_name in public_dataset.get_all_function_name():
        get_and_plot_public_function_trace(start_day, end_day, data_type, function_name, resolution)


if __name__ == "__main__":
    plot_all_trace_private()
    print("Done!")
    plot_all_trace_public()
    print("Done!")

