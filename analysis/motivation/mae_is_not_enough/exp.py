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



from py_plotter.plot import Plotter

# 从私有数据集中获取数据
private_dataset = HuaweiPrivateDataset()
public_dataset = HuaweiPublicDataset()
my_plotter = Plotter(
    figsize=(12,6),
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





if __name__ == "__main__":
    compare_prediction_metrics()


