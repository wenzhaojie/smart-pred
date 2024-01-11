from numpy import random
from datetime import datetime
import pandas as pd


def generate_invocation_in_second_from_invocation_in_min(invocations_in_min, random_seed=0):
    # 将分钟级调用数据 --> 秒级调用数据 （用泊松分布来生成）
    random.seed(random_seed)
    invocations_in_sec = []
    for each_min in invocations_in_min:
        while True:
            lam = each_min / 60
            res = random.poisson(lam=lam, size=60)
            if sum(res) == each_min:
                break
        invocations_in_sec.extend(res)
    return invocations_in_sec

def test_generate_invocation_in_second_from_invocation_in_min():
    print(f"开始测试 generate_invocation_in_second_from_invocation_in_min")
    invocations_in_min = [60, 120, 180, 240, 300]
    invocations_in_sec = generate_invocation_in_second_from_invocation_in_min(invocations_in_min)
    print(invocations_in_sec)


def get_continuous_date(num, interval="day", base_format="2000-01-01 00:00:00"):
    # 获取连续的日期
    date_base = datetime.fromisoformat(base_format)
    date_base_timestamp = date_base.timestamp()

    date_list = []
    if interval == "day":
        stamp_gap = 60 * 60 * 24
    elif interval == "hour":
        stamp_gap = 60 * 60
    elif interval == "min":
        stamp_gap = 60
    else:
        print(f"interval必须是 day,hour,min 中的一种")
        stamp_gap = 60
    for index in range(num):
        date = datetime.fromtimestamp(date_base_timestamp + index * stamp_gap)
        date_list.append(date)
    return date_list


def test_get_continuous_date():
    print(f"开始测试 get_continuous_date")
    date_list = get_continuous_date(10, interval="day", base_format="2000-01-01 00:00:00")
    print(date_list)



def cal_diff(trace):
    # 计算差分序列, 其中第一个差分值为0
    df = pd.DataFrame({"data":trace})
    res = df.diff()["data"].values.tolist()
    res[0] = 0
    return res


def test_cal_diff():
    print(f"开始测试 cal_diff")
    trace = [1, 2, 3, 4, 5]
    res = cal_diff(trace)
    print(res)



def is_simple_period(trace, threshold=5):
    # 判断某一个调用序列是否是简单周期性
    last_invocation = -1
    iat_list = []
    for index, invocation_num in enumerate(trace):
        if invocation_num >= 1:
            if last_invocation == -1:
                last_invocation = index
                continue # 跳过第一次调用
            # 有调用时记录IAT
            iat_list.append(index - last_invocation)
            last_invocation = index

    if len(iat_list) < 2:
        print("序列太短，无法判断周期性。")
        return False

    def percentile(data, percentile):
        # 计算百分位数
        index = int(len(data) * percentile / 100)
        return data[index]

    # 计算IAT的P90和P10
    iat_list.sort()
    p90 = percentile(iat_list, 90)
    p10 = percentile(iat_list, 10)

    # 判断是否是简单周期性
    delta = p90 - p10
    print(f"P90与P10的差值为: {delta}")
    if delta <= threshold:
        print("是简单周期!")
        return True
    else:
        print("不是简单周期!")
        return False




def test_is_simple_period():
    print(f"开始测试 is_simple_period")
    trace = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    is_simple_period(trace, threshold=1)
    trace = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    is_simple_period(trace, threshold=1)


def test_trace(trace, idle_rate_range=(0,0.5), avg_rate_range=(1,10), max_range=(0,100)):
    # 过滤trace
    # idle_rate_range: 空闲率范围，指的是trace中为0的比例
    # avg_rate_range: 平均利用率范围，指的是trace中所有值的平均值
    # max_range: 最大值范围，指的是trace中的最大值
    # 我们要求trace中的空闲率在idle_rate_range范围内，平均利用率在avg_rate_range范围内，最大值在max_range范围内
    # 如果满足条件，返回True，否则返回False

    # 计算空闲率，即 trace 中 0 的比例
    idle_rate = trace.count(0) / len(trace) if trace else 0

    # 计算平均利用率
    avg_rate = sum(trace) / len(trace) if trace else 0

    # 获取最大值
    max_val = max(trace) if trace else 0

    # 打印 trace 的统计信息
    print(f"idle_rate: {idle_rate}")
    print(f"avg_rate: {avg_rate}")
    print(f"max_val: {max_val}")

    # 检查 trace 是否满足所有有效条件
    conditions = []
    if idle_rate_range is not None:
        conditions.append(idle_rate_range[0] <= idle_rate <= idle_rate_range[1])
    if avg_rate_range is not None:
        conditions.append(avg_rate_range[0] <= avg_rate <= avg_rate_range[1])
    if max_range is not None:
        conditions.append(max_range[0] <= max_val <= max_range[1])

    return all(conditions)


def test_test_trace():
    trace = [1,0,3,4,5,6]
    res = test_trace(trace, idle_rate_range=(0, 0.5), avg_rate_range=(1, 10), max_range=(0, 100))
    print(res)


if __name__ == "__main__":
    test_generate_invocation_in_second_from_invocation_in_min()
    test_get_continuous_date()
    test_cal_diff()
    test_test_trace()
    test_is_simple_period()
