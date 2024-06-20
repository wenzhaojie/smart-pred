from smart_pred.model.local.histogram import Histogram_model, OpenFaaS_model
from smart_pred.dataset.crane_trace import CraneDataset
from smart_pred.dataset.azure_trace_2019 import AzureFunction2019

def plot_a_figure_hist(function_name="9"):
    extra_parameters = {
        "seq_len": 1440 * 5,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    }

    # # 获取Crane数据
    # ds = CraneDataset()
    # trace = ds.get_data_by_day_range(start_day=0, end_day=5, data_type="requests",
    #                                  function_name=function_name, resolution="minute")

    # 获取稀疏的数据
    ds = AzureFunction2019()
    ds.load_and_cache_dataset()

    trace = ds.get_all_invocation_trace_by_hash_function(hash_function="1b0736fd9c51899497cfa21385cabab7dd9c37f64af4234eed16c4c48edf8b64")


    # 去除NaN值
    for i in range(len(trace)):
        if trace[i] != trace[i]:  # 判断是否为NaN
            trace[i] = 0

    print("trace:", trace)

    # 前3天训练，预测第4天
    history = trace[:-1440]
    true = trace[-1440:]

    # 预测下一个周期的点
    model = Histogram_model()
    log_dict, prediction = model.use_future_rolling_evaluation(history, test=true, extra_parameters=extra_parameters)
    print("下一个周期的预测值:", prediction)
    print("下一个周期的真实值:", true)

    # 画图
    from py_plotter.plot import Plotter

    my_plotter = Plotter(
        font_thirdparty="YaHei",
        dpi=300,
        figsize=(10, 6),
    )

    x = list(range(len(true)))

    file_name = f"Histogram_model_{function_name}.pdf"
    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[true, prediction],
        legend_label_list=["true", "pred"],
        title="Histogram Model",
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        y_grid=True,
        x_grid=True,
        y_tick_ndigits=0,
        x_tick_ndigits=0,
        filename=file_name
    )


def plot_a_figure_OpenFaaS(function_name="9"):
    extra_parameters = {
        "seq_len": 1440 * 5,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
    }

    # # 获取Crane数据
    # ds = CraneDataset()
    # trace = ds.get_data_by_day_range(start_day=0, end_day=5, data_type="requests",
    #                                  function_name=function_name, resolution="minute")

    # 获取稀疏的数据
    ds = AzureFunction2019()
    ds.load_and_cache_dataset()

    trace = ds.get_all_invocation_trace_by_hash_function(
        hash_function="1b0736fd9c51899497cfa21385cabab7dd9c37f64af4234eed16c4c48edf8b64"
    )

    # 去除NaN值
    for i in range(len(trace)):
        if trace[i] != trace[i]:  # 判断是否为NaN
            trace[i] = 0

    print("trace:", trace)

    # 前3天训练，预测第4天
    history = trace[:-1440]
    true = trace[-1440:]

    # 预测下一个周期的点
    model = OpenFaaS_model()
    log_dict, prediction = model.use_future_rolling_evaluation(history, test=true, extra_parameters=extra_parameters)
    print("下一个周期的预测值:", prediction)

    # 画图
    from py_plotter.plot import Plotter

    my_plotter = Plotter(
        font_thirdparty="YaHei",
        dpi=300,
        figsize=(10, 6),
    )

    x = list(range(len(true)))

    file_name = f"OpenFaaS_model_{function_name}.pdf"
    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[true, prediction],
        legend_label_list=["true", "pred"],
        title="OpenFaaS Model",
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        y_grid=True,
        x_grid=True,
        y_tick_ndigits=0,
        x_tick_ndigits=0,
        filename=file_name
    )



if __name__ == "__main__":
    plot_a_figure_hist(function_name="9")
    plot_a_figure_OpenFaaS(function_name="9")

