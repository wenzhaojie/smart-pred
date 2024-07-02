# 画图来展示不同数据集的平均预测性能
from py_plotter.plot import Plotter
import pandas as pd

my_plotter = Plotter(
    figsize=(10,8),
    dpi=300,
    font_thirdparty="YaHei"
)


# 平均几个csv中指标的值
def avg_metric_from_csv_files(csv_filepath_list=None, avg_csv_filename=None, filter_method_list=None,
                              filter_metric_list=None):
    if csv_filepath_list is None or avg_csv_filename is None or filter_method_list is None or filter_metric_list is None:
        print(
            "Please provide all necessary inputs: csv_filepath_list, avg_csv_filename, filter_method_list, and filter_metric_list")
        return

    # Initialize a DataFrame to store the sum of metrics
    metrics_sum = pd.DataFrame(columns=['method'] + filter_metric_list)
    metrics_count = pd.DataFrame(columns=['method'] + filter_metric_list)

    metrics_sum['method'] = filter_method_list
    metrics_count['method'] = filter_method_list

    for metric in filter_metric_list:
        metrics_sum[metric] = 0.0
        metrics_count[metric] = 0

    for file_path in csv_filepath_list:
        df = pd.read_csv(file_path)

        # Filter the methods and metrics
        df_filtered = df[df['method'].isin(filter_method_list)]
        df_filtered = df_filtered[['method'] + filter_metric_list]

        for method in filter_method_list:
            method_df = df_filtered[df_filtered['method'] == method]
            if not method_df.empty:
                metrics_sum.loc[metrics_sum['method'] == method, filter_metric_list] += method_df[
                    filter_metric_list].astype(float).values
                metrics_count.loc[metrics_count['method'] == method, filter_metric_list] += 1

    # Calculate the average
    metrics_avg = metrics_sum.copy()
    for method in filter_method_list:
        metrics_avg.loc[metrics_avg['method'] == method, filter_metric_list] /= metrics_count.loc[
            metrics_count['method'] == method, filter_metric_list]

    # Save the result to a new CSV file
    metrics_avg.to_csv(avg_csv_filename, index=False)

    print(f"Averages saved to {avg_csv_filename}")



def merge_crane_dataset():
    csv_files = [
        'results/CraneDataset_8_0_4/experiment_parameters.csv',
        'results/CraneDataset_9_0_4/experiment_parameters.csv',
        'results/CraneDataset_11_0_4/experiment_parameters.csv',
        'results/CraneDataset_14_0_4/experiment_parameters.csv',
    ]
    filter_methods = ['SmartPred', 'NHITS', 'NBEATS', 'PatchTST', "DLinear", "Quantile", "Movingavg", "Avgvalue",]
    filter_metrics = ['cold_start_invocation_ratio', 'utilization_ratio', 'mae', 'rmse', 'mape',
                      'selective_asymmetric_mae', ]

    avg_metric_from_csv_files(
        csv_filepath_list=csv_files,
        avg_csv_filename='avg_crane.csv',
        filter_method_list=filter_methods,
        filter_metric_list=filter_metrics
    )


def merge_huawei_dataset():
    csv_files = [
        'results/HuaweiPrivateDataset_4_0_4/experiment_parameters.csv',
        'results/HuaweiPrivateDataset_28_0_4/experiment_parameters.csv',
        'results/HuaweiPublicDataset_4_0_4/experiment_parameters.csv',
    ]
    filter_methods = ['SmartPred', 'NHITS', 'NBEATS', 'PatchTST', "DLinear", ]
    filter_metrics = ['cold_start_invocation_ratio', 'utilization_ratio', 'mae', 'rmse', 'mape',
                      'selective_asymmetric_mae', ]

    avg_metric_from_csv_files(
        csv_filepath_list=csv_files,
        avg_csv_filename='avg_huawei.csv',
        filter_method_list=filter_methods,
        filter_metric_list=filter_metrics
    )


def merge_azure_dataset():
    csv_files = [
        "results/AzureFunction2019_1ad3a3335bb2c127474fbfd71ef278319ad4c58ed0bfe4d5e152f77d18ebbe33_0_4/experiment_parameters.csv",
        "results/AzureFunction2019_94a32ea3f04599336f20deb4aa3ebf10722c90ec0a3e0da8834f171b91287487_0_4/experiment_parameters.csv",
        'results/AzureFunction2019_9c9e6b50fe3a4cd0fad2bd44a0b01d6775f0f766b68f756663ed677c7fa6ec70_0_4/experiment_parameters.csv',
    ]
    filter_methods = ['SmartPred', 'NHITS', 'NBEATS', 'PatchTST', "DLinear", ]
    filter_metrics = ['cold_start_invocation_ratio', 'utilization_ratio', 'mae', 'rmse', 'mape', 'selective_asymmetric_mae',]

    avg_metric_from_csv_files(
        csv_filepath_list=csv_files,
        avg_csv_filename='avg_azure.csv',
        filter_method_list=filter_methods,
        filter_metric_list=filter_metrics
    )




if __name__ == "__main__":
    merge_crane_dataset()
    merge_huawei_dataset()
    merge_azure_dataset()