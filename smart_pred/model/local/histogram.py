from smart_pred.model.local.base import Basic_model
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import time
from smart_pred.utils.metrics import get_metric_dict


# 返回cdf中指定分位数之前的最后bucket的位置（分钟）
def find_precentile(cdf, percent, head=False):
    """ Returns the last whole bucket (minute) before the percentile """
    for i, value in enumerate(cdf):
        if percent < value:
            if head:
                return max(0, i-1)
            else:
                return min(i+1, len(cdf))
    return len(cdf)


class Histogram_model(Basic_model):
    def __init__(self, name="Histogram_model", scaler=MinMaxScaler()):
        print(f"初始化 Histogram Model!")
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.lower_threshold = 0.05
        self.higher_threshold = 0.95
        self.default_extra_parameters = {
            "seq_len": 240,
            "pred_len": 720,
            "is_scaler": False,
            "is_round": True,

        }
        pass

    # 用于计算 prewarm_time 和 keepalive_time
    def cal_windows(self, history):
        # 先将history转换为 list
        history = list(history)
        # 进行IAT的转换
        last_invoked = 0
        iat_list = []
        for i in range(len(history)):
            if history[i] >= 1:
                # 有调用
                iat = i - last_invoked
                iat_list.append(iat)
                last_invoked = i
            else:
                continue
        hist, bin_edges = np.histogram(iat_list, bins=len(iat_list))
        # 统计cdf
        cdf = np.cumsum(hist / hist.sum())
        head = find_precentile(cdf, self.lower_threshold, head=True)
        tail = find_precentile(cdf, self.higher_threshold)
        # 得出 prewarm_time 和 keepalive_time
        prewarm_time = bin_edges[head]
        keepalive_time = bin_edges[tail - head]

        return prewarm_time, keepalive_time

    def use_future_rolling_evaluation(self, train, test, extra_parameters=None, plotter=None):
        '''
        用于评估模型的效果
        train: 训练数据
        test: 预测真实的数据
        is_scaler: 是否使用归一化
        use_future: 是否用真实的值来滚动测试性能
        is_round: 是否将预测的连续结果转换成整数
        '''
        is_round = extra_parameters["is_round"]
        is_scaler = extra_parameters["is_scaler"]

        # histogram 不做归一化
        # 先做归一化
        if is_scaler == True:
            combined_trace = np.concatenate((np.array(train), np.array(test)))
            processed_trace = self.scaler.fit_transform(combined_trace.reshape(-1, 1)).reshape(-1, )
            train = processed_trace[:len(train)]
            test = processed_trace[len(train):]
        else:
            train = np.array(train)
            test = np.array(test)

        train = list(train)
        test = list(test)
        history_add = list(test)

        # 获得预测参数
        seq_len = extra_parameters["seq_len"]
        pred_len = extra_parameters["pred_len"]

        history = train
        predict_list = []
        compute_t_list = []
        pointer = 0

        warm_range = None  # 用于保存缓存的预测结果(起点，终点)相对于 test
        last_concurrency = 1  # 用于记录上一次的concurrency，初始为1

        # print(f"开始滚动预测")
        # 滚动预测
        start_t = time.time()
        # 模拟时钟
        while len(predict_list) < len(test):

            # 获取 input
            if pointer < seq_len:
                history_base = history[-seq_len + len(predict_list):]
                history_seq = history_base + history_add[:pointer]
            else:
                history_base = []
                history_seq = history_base + history_add[pointer - seq_len:pointer]

            if warm_range is None:  # 没有预测的值

                # 进行预测
                prewarm_time, keepalive_time = self.cal_windows(history=history_seq)
                warm_range = (pointer + prewarm_time, pointer + prewarm_time + keepalive_time)
                print(f"warm_range:{warm_range}")

            # 查看当前的predict值
            if pointer >= warm_range[0] and pointer <= warm_range[1]:
                print(f"落入缓存范围")
                predict = last_concurrency  # 预测值为上一次的调用
            else:
                predict = 0

            predict_list.append(predict)

            # 有调用，更新warm_range和last_concurrency
            # print(f"有调用，更新warm_range和last_concurrency")
            if test[pointer] > 0:  # 如果有调用
                # print(f"如果有调用")
                prewarm_time, keepalive_time = self.cal_windows(history=history_seq)
                warm_range = (pointer + prewarm_time, pointer + prewarm_time + keepalive_time)
                last_concurrency = test[pointer]
                print(f"有调用，更新warm_range和last_concurrency:{warm_range},{last_concurrency}")

            pointer += 1

        predict_t = time.time() - start_t  # 计算时间
        # predict 转换为 numpy
        predict = np.array(predict_list)
        # test 转换为 numpy
        test = np.array(test)

        # 做还原归一化
        if is_scaler == True:
            predict = self.scaler.inverse_transform(predict.reshape(-1, 1)).reshape(-1, )
            test = self.scaler.inverse_transform(test.reshape(-1, 1)).reshape(-1, )

        # 指标计算
        metrics_dict = get_metric_dict(y_pred=predict_list, y_test=test)

        # 收集日志
        log_dict = {
            "model": self.name,
            "train_length": len(train),
            "test_length": len(test),
            "predict_t": predict_t,
            "train_t": 0
        }
        log_dict.update(metrics_dict)
        print(f"log:{log_dict}")

        return log_dict, predict_list


class OpenFaaS_model(Histogram_model):

    def __init__(self, name="OpenFaaS_model", scaler=MinMaxScaler()):
        print(f"初始化 OpenFaaS_model!")
        super().__init__(name, scaler)
        self.name = name
        self.model = None
        self.scaler = scaler
        self.default_extra_parameters = {
            "seq_len": 1440 * 3,
            "pred_len": 1440,
            "moving_window": 20,
            "history_error_correct": False,
            "is_scaler": False,
            "use_future": True,
            "is_round": True,
        }
        # print(f"self.scaler:{self.scaler}")
        pass

    def cal_windows(self, history):
        # 先将history转换为 list
        prewarm_time = 0
        keepalive_time = 15

        return prewarm_time, keepalive_time
