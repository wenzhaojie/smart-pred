import numpy as np
import statsmodels.api as sm


def predict_periodic_series(history, period, quantile=0.5):
    """
    预测具有周期性的时间序列数据

    参数：
    history (list): 历史数据，一个具有周期性的时间序列
    period (int): 周期长度
    quantile (float): 分位数，默认为0.5

    返回值：
    prediction (float): 下一个周期的预测值
    """
    n = len(history)
    predictions = []

    # 遍历不同的相位
    for i in range(period):
        # 从历史数据中选取相同相位的数据
        data = [history[j] for j in range(i, n, period)]
        # 构建自变量
        X = np.arange(len(data))
        # 构建分位数回归模型并拟合
        model = sm.QuantReg(data, sm.add_constant(X)).fit(q=quantile)
        # 预测下一个周期的点
        X_pred = np.arange(len(data)+1)
        next_prediction = model.predict(sm.add_constant(X_pred))

        print(f"next_prediction:{next_prediction}")
        # 最后一个值即为预测值
        pred_value = next_prediction[-1]
        predictions.append(pred_value)

    # 返回预测值
    return predictions

if __name__ == '__main__':
    # 示例历史数据
    history = [15, 25, 35, 45, 11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43, 14, 24, 34, 44]
    # 周期长度
    period = 4

    # 预测下一个周期的点
    prediction = predict_periodic_series(history, period, quantile=0.99)
    print("下一个周期的预测值:", prediction)
