import math  # 导入math库
import numpy as np  # 导入numpy库，并命名为np
from sklearn import metrics  # 从sklearn库中导入metrics模块
from sklearn.metrics import mean_squared_error  # 从sklearn.metrics中导入mean_squared_error函数
from sklearn.metrics import r2_score  # 从sklearn.metrics中导入r2_score函数


# 定义计算平均平方误差的函数
def mse(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)  # 使用sklearn的mean_squared_error计算MSE
    return mse

# 定义计算均方根误差的函数
def rmse(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)  # 首先计算MSE
    rmse = math.sqrt(mse)  # 计算MSE的平方根得到RMSE
    return rmse

# 定义计算R2分数的函数
def r2(y_pred, y_test):
    r2 = r2_score(y_test, y_pred)  # 使用sklearn的r2_score函数计算R2分数
    return r2

# 定义计算平均绝对误差的函数
def mae(y_pred, y_test):
    mae = metrics.mean_absolute_error(y_pred, y_test)  # 使用sklearn的mean_absolute_error函数计算MAE
    return mae

# 自定义计算MAPE的函数
def mape(y_pred, y_test):
    y_pred = np.array(y_pred)  # 将预测值转换为numpy数组
    y_test = np.array(y_test)  # 将实际值转换为numpy数组
    return np.mean(np.abs((y_pred - y_test) / y_test)) * 100  # 计算MAPE

# 自定义计算SMAPE的函数
def smape(y_pred, y_test):
    return 2.0 * np.mean(np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) * 100  # 计算SMAPE

# 定义一个误差放大器类
class Crane_error:
    def __init__(self):
        pass

    @staticmethod
    def amplify(x):
        res = -math.log(1.0 - x) / math.log(1.25)  # 使用数学公式放大误差
        return res

    @staticmethod
    def MAPE(actual, predicted):
        epsilon = 1e-3  # 设置一个小的正数以避免除以零的错误
        e = 0
        assert (len(actual) == len(predicted))  # 确保实际值和预测值长度相同
        for (act, pred) in zip(actual, predicted):
            if act < epsilon:
                return "Error"  # 如果实际值太小，则返回错误
            if pred < act:
                # 如果预测值小于实际值，放大误差
                e += Crane_error.amplify((act - pred) / act)
            else:
                # 否则直接计算误差
                e += (pred - act) / act
        e = e / float(len(actual))  # 计算平均误差
        return e

    @staticmethod
    def PredictionError(y_pred, y_test):
        mape = Crane_error.MAPE(y_test, y_pred)  # 使用MAPE计算误差
        if mape == "Error":
            return mae(y_pred, y_test)  # 如果MAPE返回错误，则使用MAE
        else:
            return mape

# 计算cold_start_ratio的函数
def cold_start_invocation_ratio(y_pred, y_test):
    assert len(y_pred) == len(y_test)  # 确保预测值和实际值长度相同
    cold_start_invocation_count = 0
    for pred, test in zip(y_pred, y_test):
        if test > pred:
            cold_start_invocation_count += test - pred
    _cold_start_ratio = cold_start_invocation_count / sum(y_test)
    return _cold_start_ratio

# 计算utilization_ratio的函数
def utilization_ratio(y_pred, y_test):
    assert len(y_pred) == len(y_test)  # 确保预测值和实际值长度相同
    total_res = 0
    utilized_res = 0
    for pred, test in zip(y_pred, y_test):
        # 如果 pred > test，则说明预测值大于实际值，即资源存在浪费
        if pred > test:
            utilized_res += test
            total_res += pred
        else:
            # 如果 pred <= test，则说明预测值小于等于实际值，即资源被充分利用
            utilized_res += test
            total_res += test
    _utilization_ratio = utilized_res / total_res  # 计算utilization_ratio
    return _utilization_ratio

# 计算over_provisioned_ratio的函数
def over_provisioned_ratio(y_pred, y_test):
    assert len(y_pred) == len(y_test)  # 确保预测值和实际值长度相同
    over_provisioned_count = 0
    for pred, test in zip(y_pred, y_test):
        if pred > test:
            over_provisioned_count += pred - test
    _over_provisioned_ratio = over_provisioned_count / sum(y_test)  # 计算over_provisioned_ratio
    return _over_provisioned_ratio

# 计算cold_start_time_slot_ratio的函数
def cold_start_time_slot_ratio(y_pred, y_test):
    assert len(y_pred) == len(y_test)  # 确保预测值和实际值长度相同
    cold_start_time_slot_count = 0
    for pred, test in zip(y_pred, y_test):
        if pred < test:
            cold_start_time_slot_count += 1
    _cold_start_time_slot_ratio = cold_start_time_slot_count / len(y_test)  # 计算cold_start_time_slot_ratio
    return _cold_start_time_slot_ratio


def selective_asymmetric_loss_mse(y_true, y_pred, alpha=0.95, penalty_factor=2.0, high_penalty_factor=3.0):
    """
    选择性非对称损失函数，对真实值（负载）变化量位于顶部alpha分位的时间点，如果预测值低估了真实值，则施加更高的惩罚。

    :param y_true: 真实值数组（每个时间点的负载）
    :param y_pred: 预测值数组
    :param alpha: 分位数阈值，用于选择高惩罚时间点
    :param penalty_factor: 默认的低估惩罚因子
    :param high_penalty_factor: 选定时间点的增加惩罚因子
    :return: 计算出的损失
    """
    # 计算真实值（负载）的变化量
    load_changes = np.diff(y_true, prepend=y_true[0])  # 使用prepend保持数组尺寸一致

    # 寻找负载变化量位于顶部alpha分位的阈值
    threshold = np.percentile(load_changes, 100 * alpha)

    # 确定哪些时间点属于高惩罚组
    high_penalty_indices = load_changes >= threshold

    # 计算残差
    residuals = y_true - y_pred

    # 应用惩罚
    losses = np.where(
        residuals > 0,
        np.where(high_penalty_indices, high_penalty_factor * (residuals ** 2),penalty_factor * (residuals ** 2)),
        residuals ** 2
    )

    return np.mean(losses)



def selective_asymmetric_loss_mae(y_true, y_pred, alpha=0.95, penalty_factor=2.0, high_penalty_factor=3.0):
    """
    选择性非对称损失函数，对真实值（负载）变化量位于顶部alpha分位的时间点，如果预测值低估了真实值，则施加更高的惩罚。
    这一版本的损失函数基于平均绝对误差（MAE）。

    :param y_true: 真实值数组（每个时间点的负载）
    :param y_pred: 预测值数组
    :param alpha: 分位数阈值，用于选择高惩罚时间点
    :param penalty_factor: 默认的低估惩罚因子
    :param high_penalty_factor: 选定时间点的增加惩罚因子
    :return: 计算出的损失
    """
    # 计算真实值（负载）的变化量
    load_changes = np.diff(y_true, prepend=y_true[0])  # 使用prepend保持数组尺寸一致

    # 寻找负载变化量位于顶部alpha分位的阈值
    threshold = np.percentile(load_changes, 100 * alpha)

    # 确定哪些时间点属于高惩罚组
    high_penalty_indices = load_changes >= threshold

    # 计算残差的绝对值
    residuals = np.abs(y_true - y_pred)

    # 应用惩罚
    losses = np.where(y_true > y_pred,
                      np.where(high_penalty_indices, high_penalty_factor * residuals, penalty_factor * residuals),
                      residuals)

    return np.mean(losses)


def selective_asymmetric_sample_loss_mse(y_true, y_pred, alpha=0.95, penalty_factor=2.0, high_penalty_factor=3.0):
    """
    选择性非对称损失函数，对真实值（负载）变化量位于顶部alpha分位的时间点，如果预测值低估了真实值，则施加更高的惩罚。

    :param y_true: 真实值数组（每个时间点的负载）
    :param y_pred: 预测值数组
    :param alpha: 分位数阈值，用于选择高惩罚时间点
    :param penalty_factor: 默认的低估惩罚因子
    :param high_penalty_factor: 选定时间点的增加惩罚因子
    :return: 计算出的损失，是一个list
    """
    # 计算真实值（负载）的变化量
    load_changes = np.diff(y_true, prepend=y_true[0])  # 使用prepend保持数组尺寸一致

    # 寻找负载变化量位于顶部alpha分位的阈值
    threshold = np.percentile(load_changes, 100 * alpha)

    # 确定哪些时间点属于高惩罚组
    high_penalty_indices = load_changes >= threshold

    # 计算残差
    residuals = y_true - y_pred

    # 应用惩罚
    losses = np.where(
        residuals > 0,
        np.where(high_penalty_indices, high_penalty_factor * (residuals ** 2),penalty_factor * (residuals ** 2)),
        residuals ** 2
    )

    return losses



def selective_asymmetric_sample_loss_mae(y_true, y_pred, alpha=0.95, penalty_factor=2.0, high_penalty_factor=3.0):
    """
    选择性非对称损失函数，对真实值（负载）变化量位于顶部alpha分位的时间点，如果预测值低估了真实值，则施加更高的惩罚。
    这一版本的损失函数基于平均绝对误差（MAE）。

    :param y_true: 真实值数组（每个时间点的负载）
    :param y_pred: 预测值数组
    :param alpha: 分位数阈值，用于选择高惩罚时间点
    :param penalty_factor: 默认的低估惩罚因子
    :param high_penalty_factor: 选定时间点的增加惩罚因子
    :return: 计算出的损失，是一个list
    """
    # 计算真实值（负载）的变化量
    load_changes = np.diff(y_true, prepend=y_true[0])  # 使用prepend保持数组尺寸一致

    # 寻找负载变化量位于顶部alpha分位的阈值
    threshold = np.percentile(load_changes, 100 * alpha)

    # 确定哪些时间点属于高惩罚组
    high_penalty_indices = load_changes >= threshold

    # 计算残差的绝对值
    residuals = np.abs(y_true - y_pred)

    # 应用惩罚
    losses = np.where(y_true > y_pred,
                      np.where(high_penalty_indices, high_penalty_factor * residuals, penalty_factor * residuals),
                      residuals)

    return losses


# 计算并返回一个包含多个性能指标的字典
def get_metric_dict(y_pred, y_test):
    _rmse = rmse(y_pred=y_pred, y_test=y_test)  # 计算RMSE
    _mse = mse(y_pred=y_pred, y_test=y_test)  # 计算MSE
    _r2 = r2(y_pred=y_pred, y_test=y_test)  # 计算R2
    # MAE
    _mae = mae(y_pred=y_pred, y_test=y_test)  # 计算MAE
    _cold_start_invocation_ratio = cold_start_invocation_ratio(y_pred=y_pred, y_test=y_test)  # 计算cold_start_ratio
    _utilization_ratio = utilization_ratio(y_pred=y_pred, y_test=y_test)  # 计算utilization_ratio
    _over_provisioned_ratio = over_provisioned_ratio(y_pred=y_pred, y_test=y_test)  # 计算over_provisioned_ratio
    _cold_start_time_slot_ratio = cold_start_time_slot_ratio(y_pred=y_pred, y_test=y_test)  # 计算cold_start_time_slot_ratio
    _selective_asymmetric_loss_mse = selective_asymmetric_loss_mse(y_true=y_test, y_pred=y_pred)  # 计算选择性非对称损失
    _selective_asymmetric_loss_mae = selective_asymmetric_loss_mae(y_true=y_test, y_pred=y_pred)  # 计算选择性非对称损失（基于MAE）

    try:
        _crane_error = Crane_error().PredictionError(y_pred=y_pred, y_test=y_test)  # 尝试计算Crane_error
    except Exception:
        _crane_error = None  # 如果计算失败，设置为None
        pass

    metrics_dict = {
        "rmse": _rmse,
        "mse": _mse,
        "r2": _r2,
        "mae": _mae,
        "crane_error": _crane_error,
        "cold_start_invocation_ratio": _cold_start_invocation_ratio,
        "utilization_ratio": _utilization_ratio,
        "over_provisioned_ratio": _over_provisioned_ratio,
        "cold_start_time_slot_ratio": _cold_start_time_slot_ratio,
        "selective_asymmetric_loss_mse": _selective_asymmetric_loss_mse,
        "selective_asymmetric_loss_mae": _selective_asymmetric_loss_mae
    }
    return metrics_dict


def test_get_metric_dict():
    print("开始测试 get_metric_dict")
    y_test = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
    y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])

    metric_dict = get_metric_dict(y_pred, y_test)
    print(metric_dict)


# 测试只有一个样本的
def test_sample_loss():
    print("开始测试 sample_loss")
    y_test = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
    y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])

    # sample loss
    _selective_asymmetric_sample_loss_mse = selective_asymmetric_sample_loss_mse(
        y_true=y_test,
        y_pred=y_pred
    )
    print(f"selective_asymmetric_sample_loss_mse: {_selective_asymmetric_sample_loss_mse}")


if __name__ == "__main__":
    # test_get_metric_dict()
    test_sample_loss()