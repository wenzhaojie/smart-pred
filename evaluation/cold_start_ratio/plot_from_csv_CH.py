from py_plotter.plot import Plotter
import pandas as pd


my_plotter = Plotter(
    figsize=(10,8),
    font_thirdparty="YaHei",
    fontsize=30,
)



def plot_figures(csv_filepath="./result.csv"):
    # 读取csv
    df = pd.read_csv(filepath_or_buffer=csv_filepath)

    # legend_label_list
    legend_label_list = ["Histogram", "IceBreaker", "KeepAlive", "SmartPred"]
    #


    print(df.head())




if __name__ == "__main__":
    plot_figures()