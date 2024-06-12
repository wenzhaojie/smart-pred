import torch

def selective_asymmetric_loss_mae(y_true, y_pred, alpha=0.95, penalty_factor=2.0, high_penalty_factor=3.0):
    """
    选择性非对称损失函数，对真实值（负载）变化量位于顶部alpha分位的时间点，如果预测值低估了真实值，则施加更高的惩罚。
    这一版本的损失函数基于平均绝对误差（MAE）。

    :param y_true: 真实值张量（每个时间点的负载）
    :param y_pred: 预测值张量
    :param alpha: 分位数阈值，用于选择高惩罚时间点
    :param penalty_factor: 默认的低估惩罚因子
    :param high_penalty_factor: 选定时间点的增加惩罚因子
    :return: 计算出的损失
    """
    # 计算真实值（负载）的变化量
    load_changes = torch.diff(y_true, prepend=y_true[:1])  # 使用prepend保持张量尺寸一致

    # 寻找负载变化量位于顶部alpha分位的阈值
    threshold = torch.quantile(load_changes, alpha)

    # 确定哪些时间点属于高惩罚组
    high_penalty_indices = load_changes >= threshold

    # 计算残差的绝对值
    residuals = torch.abs(y_true - y_pred)

    # 创建一个与residuals相同形状的张量，初始值为 penalty_factor * residuals
    losses = penalty_factor * residuals

    # 对于高惩罚组的时间点，更新损失值为 high_penalty_factor * residuals
    high_penalty_mask = high_penalty_indices & (y_true > y_pred)
    losses[high_penalty_mask] = high_penalty_factor * residuals[high_penalty_mask]

    # 对于低估的时间点，应用 penalty_factor 惩罚
    low_penalty_mask = (y_true > y_pred) & ~high_penalty_mask
    losses[low_penalty_mask] = penalty_factor * residuals[low_penalty_mask]

    return torch.mean(losses)



if __name__ == '__main__':
    # 示例用法
    y_true = torch.tensor([10.0, 12.0, 15.0, 18.0, 20.0])
    y_pred = torch.tensor([9.0, 11.0, 14.0, 17.0, 21.0])

    loss = selective_asymmetric_loss_mae(y_true, y_pred)
    print(f"Calculated loss: {loss.item()}")
