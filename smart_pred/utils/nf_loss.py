import torch
from typing import Optional, Union, Tuple


# Auxiliary function to divide without NaN results
def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    div = a / b
    div[div != div] = 0.0  # NaNs to zero
    div[div == float("inf")] = 0.0  # Infs to zero
    return div


def _weighted_mean(losses, weights):
    """
    Compute weighted mean of losses per datapoint.
    """
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))


# Base class for loss functions
class BasePointLoss(torch.nn.Module):
    """
    Base class for point loss functions.

    **Parameters:**<br>
    `horizon_weight`: Tensor of size h, weight for each timestamp of the forecasting window. <br>
    `outputsize_multiplier`: Multiplier for the output size. <br>
    `output_names`: Names of the outputs. <br>
    """

    def __init__(self, horizon_weight, outputsize_multiplier, output_names):
        super(BasePointLoss, self).__init__()
        if horizon_weight is not None:
            horizon_weight = torch.Tensor(horizon_weight.flatten())
        self.horizon_weight = horizon_weight
        self.outputsize_multiplier = outputsize_multiplier
        self.output_names = output_names
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def _compute_weights(self, y, mask):
        """
        Compute final weights for each datapoint (based on all weights and all masks)
        Set horizon_weight to a ones[H] tensor if not set.
        If set, check that it has the same length as the horizon in x.
        """
        if mask is None:
            mask = torch.ones_like(y, device=y.device)

        if self.horizon_weight is None:
            self.horizon_weight = torch.ones(mask.shape[-1])
        else:
            assert mask.shape[-1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"

        weights = self.horizon_weight.clone()
        weights = torch.ones_like(mask, device=mask.device) * weights.to(mask.device)
        return weights * mask


# Selective Asymmetric MAE Loss
class SelectiveAsymmetricMAELoss(BasePointLoss):

    def __init__(self, horizon_weight=None, alpha=0.95, penalty_factor=2.0, high_penalty_factor=3.0):
        super(SelectiveAsymmetricMAELoss, self).__init__(horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""])
        self.alpha = alpha
        self.penalty_factor = penalty_factor
        self.high_penalty_factor = high_penalty_factor

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask=None):
        # Assume last dimension is the time dimension for multidimensional tensors
        time_dim = -1

        if y.dim() == 1:
            y = y.unsqueeze(0).unsqueeze(0)
            y_hat = y_hat.unsqueeze(0).unsqueeze(0)
        elif y.dim() == 2:
            y = y.unsqueeze(1)
            y_hat = y_hat.unsqueeze(1)

        load_changes = torch.diff(y, dim=time_dim, prepend=y[..., :1])
        threshold = torch.quantile(load_changes, self.alpha, dim=time_dim)
        high_penalty_indices = load_changes >= threshold.unsqueeze(time_dim)

        residuals = torch.abs(y - y_hat)
        penalties = torch.ones_like(residuals)
        penalties[y > y_hat] = self.penalty_factor

        high_penalty_mask = high_penalty_indices & (y > y_hat)
        penalties[high_penalty_mask] = self.high_penalty_factor
        weighted_residuals = residuals * penalties

        weights = self._compute_weights(y, mask)
        mean_loss = _weighted_mean(weighted_residuals, weights)

        return mean_loss  # ensure this is a tensor


class SelectiveAsymmetricMSELoss(BasePointLoss):
    def __init__(self, horizon_weight=None, alpha=0.95, penalty_factor=2.0, high_penalty_factor=3.0):
        super(SelectiveAsymmetricMSELoss, self).__init__(horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""])
        self.alpha = alpha
        self.penalty_factor = penalty_factor
        self.high_penalty_factor = high_penalty_factor

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask=None):
        time_dim = -1  # Assumes the last dimension is the time dimension

        if y.dim() == 1:
            y = y.unsqueeze(0).unsqueeze(0)
            y_hat = y_hat.unsqueeze(0).unsqueeze(0)
        elif y.dim() == 2:
            y = y.unsqueeze(1)
            y_hat = y_hat.unsqueeze(1)

        load_changes = torch.diff(y, dim=time_dim, prepend=y[..., :1])
        threshold = torch.quantile(load_changes, self.alpha, dim=time_dim)
        high_penalty_indices = load_changes >= threshold.unsqueeze(time_dim)

        residuals = (y - y_hat) ** 2  # Calculate squared differences for MSE
        penalties = torch.ones_like(residuals)
        penalties[y > y_hat] = self.penalty_factor

        high_penalty_mask = high_penalty_indices & (y > y_hat)
        penalties[high_penalty_mask] = self.high_penalty_factor
        weighted_residuals = residuals * penalties

        weights = self._compute_weights(y, mask)
        mse_loss = _weighted_mean(weighted_residuals, weights)

        return mse_loss



if __name__ == '__main__':
    # Example usage
    # Define y_true and y_pred as torch.Tensors, and possibly a mask tensor.
    y_true = torch.tensor([10.0, 20.0, 15.0, 18.0, 20.0])
    y_pred = torch.tensor([9.0, 11.0, 12.0, 21.0, 21.0])

    loss_fn = SelectiveAsymmetricMAELoss()  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MAE Loss: {loss_value}")

    y_true = torch.tensor([10.0, 20.0, 15.0, 18.0, 20.0])
    y_pred = torch.tensor([9.0, 11.0, 12.0, 21.0, 21.0])

    loss_fn = SelectiveAsymmetricMAELoss()  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MAE Loss: {loss_value}")

    y_true = torch.tensor([10.0, 20.0, 15.0, 18.0, 20.0])
    y_pred = torch.tensor([9.0, 11.0, 12.0, 21.0, 21.0])

    loss_fn = SelectiveAsymmetricMAELoss()  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MAE Loss: {loss_value}")


    # Example SelectiveAsymmetricMSELoss
    y_true = torch.tensor([10.0, 20.0, 15.0, 18.0, 20.0])
    y_pred = torch.tensor([9.0, 11.0, 12.0, 21.0, 21.0])

    loss_fn = SelectiveAsymmetricMSELoss()  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MSE Loss: {loss_value}")


