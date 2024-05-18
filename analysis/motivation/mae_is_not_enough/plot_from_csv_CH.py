# 从csv中读取数据并绘图，绘制 pred 和 true 的折线图
import pandas as pd
from py_plotter.plot import Plotter
from smart_pred.utils.metrics import get_metric_dict

my_plotter = Plotter(
    figsize=(20,8),
    fontsize=30,
    font_thirdparty="YaHei",
)


def plot_from_csv(csv_file_path, file_name):
    # 从 CSV 文件中读取数据
    df = pd.read_csv(csv_file_path)

    # 获取 x 和 y 数据
    x = df.index  # 假设时间是索引
    pred = df["pred"]
    true = df["true"]

    # 统计 cold_start_invocation_ratio
    metric_dict = get_metric_dict(pred, true)
    cold_start_invocation_ratio = metric_dict["cold_start_invocation_ratio"]

    # 使用 StandardScaler 对归一化pred
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler_pred = scaler.fit_transform(pred.values.reshape(-1, 1)).flatten()
    # 对 true 进行归一化
    scaler_true = scaler.transform(true.values.reshape(-1, 1)).flatten()

    # 统计 get_metric_dict
    metric_dict = get_metric_dict(scaler_pred, scaler_true)
    # 覆盖 cold_start_invocation_ratio
    metric_dict["cold_start_invocation_ratio"] = cold_start_invocation_ratio
    print(metric_dict)

    title = f"MAE: {metric_dict['mae']:.2f}, Cold Start Ratio: {metric_dict['cold_start_invocation_ratio'] * 100:.2f}%"

    # 使用 my_plotter.plot_lines 绘制图表
    my_plotter.plot_lines(
        x_list=[x, x],
        line_data_list=[pred, true],
        legend_label_list=["预测值", "真实值"],
        legend_title=None,
        title=title,
        x_grid=True,
        x_label="时间戳 (分钟)",
        y_label="请求数",
        save_root="./plot_trace_selected",
        filename=file_name,
        x_tick_ndigits=0,
        y_tick_ndigits=0,
    )
    print(f"已经绘制 {file_name}!")





if __name__ == "__main__":
    # 示例用法
    csv_file_path = "./plot_trace/Maxvalue_model_instance_result_noise.csv"
    file_name = "Maxvalue_model_instance_result_noise_CH.pdf"
    plot_from_csv(csv_file_path, file_name)

    # 示例用法
    csv_file_path = "./plot_trace/Movingavg_model_instance_result_moving_window_10.csv"
    file_name = "Movingavg_model_instance_result_moving_window_10_CH.pdf"
    plot_from_csv(csv_file_path, file_name)