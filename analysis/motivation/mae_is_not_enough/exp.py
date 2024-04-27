# 画出实例来说明仅仅使用MAE，MAPE这种指标是不够的

# 数据采用 private 数据集 function_name = 60

from smart_pred.dataset.huawei import HuaweiPrivateDataset
from smart_pred.dataset.huawei import HuaweiPublicDataset

# 导入model
from smart_pred.model.local.passive import Movingavg_model
from smart_pred.model.local.passive import Movingmax_model
from smart_pred.model.local.period import Maxvalue_model
from smart_pred.model.local.period import Minvalue_model
from smart_pred.model.local.period import Avgvalue_model

from smart_pred.utils.metrics import get_metric_dict



from py_plotter.plot import Plotter

# 从私有数据集中获取数据
private_dataset = HuaweiPrivateDataset()
public_dataset = HuaweiPublicDataset()
my_plotter = Plotter(
    figsize=(20,8),
    fontsize=30,
)

def compare_prediction_metrics():
    # 准备数据
    # 获取追踪数据
    trace = private_dataset.get_data_by_day_range(
        start_day=3,
        end_day=7,
        data_type="requests",
        function_name="60",
        resolution="minute"
    )
    # plot trace
    x = list(range(len(trace)))
    file_name = f"private_function_name_60_3_7.pdf"

    # 使用绘图对象绘制线图
    my_plotter.plot_lines(
        x_list=[x],
        line_data_list=[trace],
        title=None,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")

    # 现在用前3天的数据来预测第4天的数据
    history = trace[:3*24*60]
    future = trace[3*24*60:4*24*60]


    # 1.采用多个周期的平均值预测
    Avgvalue_model_instance = Avgvalue_model()
    extra_parameters = {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "moving_window": 5,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }
    log, predict = Avgvalue_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )
    print(f"Avgvalue_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Avgvalue_model_instance_result.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")

    # 2.采用多个周期的最大值预测
    Maxvalue_model_instance = Maxvalue_model()

    extra_parameters = {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "moving_window": 5,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }

    log, predict = Maxvalue_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )

    print(f"Maxvalue_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Maxvalue_model_instance_result.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )

    print(f"已经绘制 {file_name}!")

    # 2.采用多个周期的最大值预测, 增加一点平均误差
    Maxvalue_model_instance = Maxvalue_model()

    extra_parameters = {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "moving_window": 5,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }

    log, predict = Maxvalue_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )
    # 增加一个平均误差噪声，0-500 平均分布
    import numpy as np
    noise = np.random.randint(1300, 3000, len(predict))
    predict = predict + noise

    # 重新计算指标
    log = get_metric_dict(predict, future)
    print(f"Maxvalue_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Maxvalue_model_instance_result_noise.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )

    print(f"已经绘制 {file_name}!")



    # 3.采用滑动窗口的平均值预测 moving_window = 5
    moving_window = 5
    Movingavg_model_instance = Movingavg_model()
    extra_parameters = {
        "seq_len": moving_window,
        "pred_len": 1,
        "moving_window": moving_window,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }
    log, predict = Movingavg_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )
    print(f"Movingavg_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Movingavg_model_instance_result_moving_window_{moving_window}.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")

    # 3.采用滑动窗口的平均值预测 moving_window = 10
    moving_window = 10
    Movingavg_model_instance = Movingavg_model()
    extra_parameters = {
        "seq_len": moving_window,
        "pred_len": 1,
        "moving_window": moving_window,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }
    log, predict = Movingavg_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )
    print(f"Movingavg_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Movingavg_model_instance_result_moving_window_{moving_window}.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")

    # 3.采用滑动窗口的平均值预测 moving_window = 20
    moving_window = 20
    Movingavg_model_instance = Movingavg_model()
    extra_parameters = {
        "seq_len": moving_window,
        "pred_len": 1,
        "moving_window": moving_window,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }
    log, predict = Movingavg_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )
    print(f"Movingavg_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Movingavg_model_instance_result_moving_window_{moving_window}.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")

    # 3.采用滑动窗口的平均值预测 moving_window = 1
    moving_window = 1
    Movingavg_model_instance = Movingavg_model()
    extra_parameters = {
        "seq_len": moving_window,
        "pred_len": 1,
        "moving_window": moving_window,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }
    log, predict = Movingavg_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )
    print(f"Movingavg_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Movingavg_model_instance_result_moving_window_{moving_window}.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")

    # 3.采用滑动窗口的最大值预测
    Movingmax_model_instance = Movingmax_model()

    extra_parameters = {
        "seq_len": 5,
        "pred_len": 1,
        "moving_window": 5,
        "is_scaler": True,
        "is_round": False,
        "period_length": 1440,
    }

    log, predict = Movingmax_model_instance.use_future_rolling_evaluation(
        train=history,
        test=future,
        extra_parameters=extra_parameters
    )
    print(f"Movingmax_model_instance result: {log}")

    # 画图
    x = list(range(len(predict)))
    pred = predict
    true = future
    file_name = f"Movingmax_model_instance_result.pdf"

    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Ratio: {log['cold_start_invocation_ratio']:.2f}"
    # title = f"MAE: {log['mae']:.2f}, RMSE: {log['rmse']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"
    title = f"MAE: {log['mae']:.2f}, Cold Start Ratio: {log['cold_start_invocation_ratio'] * 100:.2f}%"

    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root="./plot_trace",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")


if __name__ == "__main__":
    compare_prediction_metrics()


