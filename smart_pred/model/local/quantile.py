import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from smart_pred.model.local.base import Basic_model


class Quantile_model(Basic_model):
    def __init__(self, name="Quantile_model", scaler=MinMaxScaler()):
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "is_scaler": False,
            "is_round": False,
            "quantile": 0.5,
            "period": 1440,
        }

    def predict(self, history, predict_window ,extra_parameters=None):
        history = np.array(history)
        # 如果标准化
        if extra_parameters["is_scaler"]:
            history = self.scaler.transform(history.reshape(-1, 1)).reshape(-1)
        # 如果is_round为True，则对数据进行四舍五入处理
        if extra_parameters["is_round"]:
            history = np.round(history)

        n = len(history)
        predictions = []

        period = extra_parameters["period"]
        # 遍历不同的相位
        for i in range(period):
            # 从历史数据中选取相同相位的数据
            data = [history[j] for j in range(i, n, period)]
            # 构建自变量
            X = np.arange(len(data))
            # 构建分位数回归模型并拟合
            quantile = extra_parameters["quantile"]
            model = sm.QuantReg(data, sm.add_constant(X)).fit(q=quantile)
            # 预测下一个周期的点
            X_pred = np.arange(len(data)+1)
            next_prediction = model.predict(sm.add_constant(X_pred))

            # 最后一个值即为预测值
            pred_value = next_prediction[-1]
            predictions.append(pred_value)

        # 返回预测值
        return predictions



