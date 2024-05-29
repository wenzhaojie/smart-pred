from smart_pred.model.local.quantile import Quantile_model
from smart_pred.dataset.crane_trace import CraneDataset

def plot_a_figure(quantile=0.99, function_name="9"):
    extra_parameters = {
        "seq_len": 1440 * 5,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "quantile": quantile,
        "period": 1440,
    }

    # 获取Crane数据
    ds = CraneDataset()
    trace = ds.get_data_by_day_range(start_day=0, end_day=5, data_type="requests",
                                     function_name=function_name, resolution="minute")
    # 去除NaN值
    for i in range(len(trace)):
        if trace[i] != trace[i]:  # 判断是否为NaN
            trace[i] = 0

    print("trace:", trace)

    # 前3天训练，预测第4天
    history = trace[:-1440]
    true = trace[-1440:]

    # 预测下一个周期的点
    model = Quantile_model()
    prediction = model.predict(history, predict_window=1440, extra_parameters=extra_parameters)
    print("下一个周期的预测值:", prediction)

    # 画图
    from py_plotter.plot import Plotter

    my_plotter = Plotter(
        font_thirdparty="YaHei",
        dpi=300,
        figsize=(10, 6),
    )

    x = list(range(len(true)))

    file_name = f"quantile_model_{function_name}_{quantile}.pdf"
    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[true, prediction],
        legend_label_list=["true", "pred"],
        title="Quantile Model",
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
    plot_a_figure(quantile=0.99, function_name="9")
    plot_a_figure(quantile=0.8, function_name="9")
    plot_a_figure(quantile=0.5, function_name="9")
    plot_a_figure(quantile=0.01, function_name="9")