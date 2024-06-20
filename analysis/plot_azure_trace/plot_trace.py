# 从直观的来看看crane的trace都长啥样

from smart_pred.dataset.azure_trace_2019 import AzureFunction2019

from py_plotter.plot import Plotter

# 从crane数据集中获取数据
azure_dataset = AzureFunction2019()
azure_dataset.load_and_cache_dataset()

my_plotter = Plotter(
    figsize=(30, 6),
    font_thirdparty="YaHei",
)


def get_and_plot_function_trace(dataset, start_day, end_day, function_name, filename_prefix="azure"):
    """
    获取指定函数的追踪数据并绘制图形。

    :param dataset: 数据集对象，用于获取数据
    :param start_day: 开始天数
    :param end_day: 结束天数
    :param function_name: 函数名称
    :param plotter: 用于绘图的对象
    """
    # 获取追踪数据
    trace = dataset.get_all_invocation_trace_by_hash_function(hash_function=function_name) # 14天的数据
    trace = trace[start_day * 1440:(end_day+1) * 1440]

    # 处理NaN值，将其转换为0
    for i in range(len(trace)):
        if trace[i] != trace[i]:  # 判断是否为NaN
            trace[i] = 0

    # 准备绘图数据
    x = list(range(len(trace)))
    file_name = f"{filename_prefix}_{function_name}_day_{start_day}_{end_day}.pdf"

    # 使用绘图对象绘制线图
    my_plotter.plot_lines(
        x_list=[x],
        line_data_list=[trace],
        title=f"{function_name}",
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        y_grid=True,
        x_grid=True,
        y_tick_ndigits=0,
        x_tick_ndigits=0,
        filename=file_name
    )
    print(f"已经绘制 {file_name}!")


def plot_all_azure_trace():
    """
    绘制所有crane的trace
    """
    all_function_name_list = azure_dataset.get_hash_function_set_appear_in_all_days()
    for function_name in all_function_name_list:
        get_and_plot_function_trace(azure_dataset, 0, 13, function_name, filename_prefix="azure")


def plot_first_day_azure_trace():
    """
    绘制所有crane的trace
    """
    all_function_name_list = azure_dataset.get_hash_function_set_appear_in_all_days()
    for function_name in all_function_name_list:
        get_and_plot_function_trace(azure_dataset, 0, 0, function_name, filename_prefix="azure")


def plot_second_day_azure_trace():
    """
    绘制所有crane的trace
    """
    all_function_name_list = azure_dataset.get_hash_function_set_appear_in_all_days()
    for function_name in all_function_name_list:
        get_and_plot_function_trace(azure_dataset, 1, 1, function_name, filename_prefix="azure")



if __name__ == "__main__":
    # plot_all_azure_trace()
    # print("Done!")
    plot_first_day_azure_trace()
    print("Done!")
    # plot_second_day_azure_trace()
    # print("Done!")
