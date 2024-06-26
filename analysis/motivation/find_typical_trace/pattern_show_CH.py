# 绘制一张图，体现不同类型的负载。
# 四宫格图

# 稀疏 HUAWEI public 43，51，61，97, 311, 371(密度高)，406，426（密度高）
# 连续 HUAWEI public 234, 425
# 连续 Crane request 2
# 周期 Crane request 9
# 突发 HUAWEI private 10,14,54,55,56,57,68,69,71


from smart_pred.dataset.crane_trace import CraneDataset
from smart_pred.dataset.huawei import HuaweiPublicDataset, HuaweiPrivateDataset
import pandas as pd

from py_plotter.plot import Plotter

my_plotter = Plotter(
    figsize=(10, 8),
    fontsize=30,
    font_thirdparty="YaHei",
)


def plot_type_trace(y, type, y_tick_interval=None):
    file_name = f"typical_{type}_trace_CH.pdf"
    x = list(range(len(y)))
    my_plotter.plot_lines(
        x_list=[x],
        line_data_list=[y],
        title=None,
        x_label="时间戳 (分钟)",
        y_label="请求数",
        save_root="./results",
        x_grid=True,
        y_grid=True,
        y_min=0,
        y_max=int(1.2 * max(y)),
        y_tick_interval=y_tick_interval,
        y_tick_ndigits=0,
        x_tick_ndigits=0,
        filename=file_name
    )
    print(f"已经绘制 {file_name}!")


# 画连续的
def draw_continuous():
    # Crane request 2
    dataset = CraneDataset()
    y = dataset.get_data_by_day_range(0, 0, "requests", "2", "minute")
    # 需要截取前600个
    y = y[:600]
    # 保存画图数据到csv
    pd.DataFrame(y).to_csv("continuous.csv", index=False)
    plot_type_trace(y, "continuous")

# 画稀疏的
def draw_sparse():
    # HUAWEI public 43
    dataset = HuaweiPublicDataset()
    y = dataset.get_data_by_day_range(0, 0, "requests", "43", "minute")
    # 用0替换NaN
    for i in range(len(y)):
        if y[i] != y[i]:
            y[i] = 0

    print(f"y:{y}")
    # 截取前200个
    y = y[:200]
    # 保存画图数据到csv
    pd.DataFrame(y).to_csv("sparse.csv", index=False)
    plot_type_trace(y, "sparse", y_tick_interval=1)

# 画周期性的
def draw_periodic():
    # Crane request 9
    dataset = CraneDataset()
    y = dataset.get_data_by_day_range(0, 4, "requests", "9", "minute")
    # 用0替换NaN
    for i in range(len(y)):
        if y[i] != y[i]:
            y[i] = 0

    print(f"y:{y}")

    # 保存画图数据到csv
    pd.DataFrame(y).to_csv("periodic.csv", index=False)
    plot_type_trace(y, "periodic", y_tick_interval=None)

# 画bursty的
def draw_bursty():
    # HUAWEI public 97
    dataset = HuaweiPrivateDataset()
    y = dataset.get_data_by_day_range(0, 0, "requests", "10", "minute")
    # 用0替换NaN
    for i in range(len(y)):
        if y[i] != y[i]:
            y[i] = 0

    print(f"y:{y}")
    # 截取前720个
    y = y[:360]

    # 保存画图数据到csv
    pd.DataFrame(y).to_csv("bursty.csv", index=False)
    plot_type_trace(y, "bursty", y_tick_interval=10)


if __name__ == "__main__":
    draw_continuous()
    draw_sparse()
    draw_periodic()
    draw_bursty()
