

# 华为private数据集，0-18day
class HuaweiPrivateSelected:
    def __init__(self):
        pass

    @staticmethod
    def get_period_function_name_list(self):
        # 周期性的trace
        function_name_list = [4,5,15,25,33,34,39,40,60,72,73,75,86]
        return function_name_list

    @staticmethod
    def get_non_period_sparse_function_name_list(self):
        # 非周期性稀疏的trace
        function_name_list = [6,22,38,42,43]
        return function_name_list

    @staticmethod
    def get_bursty_non_period_function_name_list(self):
        # 非周期性突发的trace
        function_name_list = [10,14,54,55,56,57,68,69,71]
        return function_name_list

    @staticmethod
    def get_bursty_change_function_name_list(self):
        # Bursty Change的trace
        function_name_list = [11,12,16,27,28,31,37,87,88]
        return function_name_list

    @staticmethod
    def get_selected_trace(self):
        function_name_list = [
            28,39,40,60,72,73,75,86,88,91,92,94,96,97,100,102,104,107,116,124,126,127,129,130,131,132,135,136,146,147,148,150,155,163,179,183,190,194,195,199
        ]
        return function_name_list


class CraneSelected:
    def __init__(self):
        pass

    @staticmethod
    def get_selected_trace(self):
        function_name_list = [
            0,2,8,9,11,12,13,14
        ]
        return function_name_list


