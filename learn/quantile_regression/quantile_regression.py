import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 创建示例时间序列数据
np.random.seed(123)
n = 100
time_index = pd.date_range(start='2022-01-01', periods=n, freq='D')
y = np.random.randn(n).cumsum()  # 随机生成累积数据
X = np.arange(n)

# 将数据转换为DataFrame格式
data = pd.DataFrame({'Date': time_index, 'y': y, 'X': X})

# 划分历史数据和测试数据
split_index = 70  # 假设前70天为历史数据，后30天为测试数据

# 定义分位数列表
quantiles = [0.1, 0.5, 0.9]

# 拟合分位数回归模型并进行预测

exog = sm.add_constant(data.iloc[:split_index]['X'])
# 打印exog
print(f"exog:{exog}")

endog = data.iloc[:split_index]['y']

predictions = []
for quantile in quantiles:
    model = sm.QuantReg(endog, exog).fit(q=quantile)
    pred = model.predict(sm.add_constant(data['X']))
    predictions.append(pred)

print(f"len pred:{len(predictions[0])}")


# 绘制预测效果图
plt.figure(figsize=(10, 6))
plt.plot(data['Date'][:split_index], data['y'][:split_index], label='Historical Data', color='blue')
plt.plot(data['Date'][split_index:], data['y'][split_index:], label='Test Data', color='red')
for i, quantile in enumerate(quantiles):
    plt.plot(data['Date'], predictions[i], label=f'Quantile {quantile}', linestyle='--')
plt.title('Quantile Regression for Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
