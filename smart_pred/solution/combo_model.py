# combo_model 负责将不同预测算法的结果进行组合，得到最终预测结果
# 假设有N个预测模型，他们都通过输入历史数据（长度为history_len），输出预测数据（长度为pred_len）。
# 此外，这N个预测模型在训练数据集上的准确率分别为train_loss1和train_loss2,...,train_lossN。
# 预测模型1的预测结果为pred1，预测模型2的预测结果为pred2,...,预测模型N的预测结果为predN。
# 然后combo_model将pred1和pred2,...,predN,进行组合，得到最终的预测结果pred。
# 具体来说，combo_model的输入为pred1和train_loss1, 还输入pred2和train_loss2, ...,还输入predN和train_lossN, 最后 combo_model的输出为pred_combo。

# 请基于pytorch实现combo_model.py中的ComboModel，并选择合适的神经网络结构，来突出一种理念：在合并多个模型输出的结果时，其历史预测的越准确，其可信的成分越高。
# 请最后帮我分析，为何你的设计符合这一理念。

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboModel(nn.Module):
    def __init__(self, num_models, history_len, pred_len):
        super(ComboModel, self).__init__()
        self.num_models = num_models  # 模型数量
        self.history_len = history_len  # 历史长度
        self.pred_len = pred_len  # 预测长度

        # 定义用于结合模型输出的神经网络层
        self.fc1 = nn.Linear(num_models * (pred_len), 128)  # 加入回测准确率，所以每个模型的输入维度加1
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, pred_len)

    def forward(self, preds, backtesting_loss):
        # 沿着最后一个维度拼接所有预测和训练损失
        combined_input = torch.cat((preds, backtesting_loss.unsqueeze(1)), dim=-1)

        # 根据模型训练损失计算权重
        weights = F.softmax(-backtesting_loss, dim=0)  # 使用负的训练损失以优先考虑较低的损失

        # 将权重应用到组合的预测上
        weighted_input = combined_input * weights.unsqueeze(1)  # 将权重扩展以匹配输入维度
        weighted_input = torch.sum(weighted_input, dim=0)  # 沿着模型维度求和

        # 进一步组合的神经网络层
        x = F.relu(self.fc1(weighted_input))
        x = F.relu(self.fc2(x))
        pred_combo = self.fc3(x)

        return pred_combo



