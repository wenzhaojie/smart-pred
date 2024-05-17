import os
import pandas as pd
import numpy as np
from py_plotter.plot import Plotter


my_plotter = Plotter(
    figsize=(20, 8),
    fontsize=30,
    font="Times New Roman",
)


def plot_pred_and_true(csv_filepath, save_root, filename, x_tick_ndigits=0, y_tick_ndigits=0, is_y_tick_sci=False):
    # 从CSV文件中读取数据
    df = pd.read_csv(csv_filepath)
    pred = df["pred"]
    true = df["true"]

    # 绘制图像
    x = np.arange(len(pred))
    x_list = [x, x]

    # 生成文件名
    file_name = f"{filename}.pdf"

    # 创建保存目录
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 将pred和true保存到csv文件中
    df.to_csv(os.path.join(save_root, f"{filename}.csv"), index=False)
    print(f"已经保存 {filename}.csv!")

    # 绘制图像
    my_plotter.plot_lines(
        x_list=x_list,
        line_data_list=[pred, true],
        legend_label_list=["Predict", "True"],
        legend_title=None,
        title=None,
        x_grid=True,
        y_grid=True,
        x_label="Time",
        y_label="Requests",
        save_root=save_root,
        filename=file_name,
        x_tick_ndigits=x_tick_ndigits,
        y_tick_ndigits=y_tick_ndigits,
        is_y_tick_sci=is_y_tick_sci,
    )
    print(f"已经绘制 {file_name}!")


if __name__ == "__main__":
    # 示例用法
    csv_filepath = "path/to/your/csv/file.csv"
    save_root = "path/to/save/directory"
    filename = "plot_pred_and_true"
    plot_pred_and_true(csv_filepath, save_root, filename)
