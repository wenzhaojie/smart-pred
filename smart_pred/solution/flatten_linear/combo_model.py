import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboModel(nn.Module):
    def __init__(self, num_models, pred_len):
        super(ComboModel, self).__init__()
        self.num_models = num_models  # 模型数量
        self.pred_len = pred_len  # 预测长度

        # 定义用于结合模型输出的神经网络层
        self.fc1 = nn.Linear(num_models * (pred_len + 1), 128)  # 加入回测准确率，所以每个模型的输入维度加1
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, pred_len)

    def forward(self, x):
        # x的维度为 (batch, num_models, pred_len+1)
        # flatten the input
        flatten_input = x.view(-1, self.num_models * (self.pred_len + 1))

        # 进一步组合的神经网络层
        x = F.relu(self.fc1(flatten_input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # activation='linear'，即不使用激活函数
        pred_combo = self.fc4(x)

        return pred_combo


class TestComboModel:
    def test_forward(self):
        num_models = 3
        pred_len = 3
        model = ComboModel(num_models, pred_len)

        # 构造输入张量
        batch_size = 2
        preds = torch.randn(batch_size, num_models, pred_len)
        backtesting_loss = torch.randn(batch_size, num_models, 1)

        # 在每一个preds的后面加上对应的backtesting_loss
        x = torch.cat((preds, backtesting_loss), dim=2)

        pred_combo = model.forward(x)

        print(f"pred_combo.shape: {pred_combo.shape}")

        print("ComboModel test passed.")


if __name__ == "__main__":
    test = TestComboModel()
    test.test_forward()
