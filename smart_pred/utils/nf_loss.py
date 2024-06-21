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
    def __init__(self, horizon_weight=None, penalty_factor=4, high_penalty_factor=8, top_quantile=0.95):
        super(SelectiveAsymmetricMAELoss, self).__init__(horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""])
        self.penalty_factor = penalty_factor
        self.high_penalty_factor = high_penalty_factor
        self.top_quantile = top_quantile

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor = None):
        under_estimation = (y_hat < y).float()  # Underestimation mask
        over_estimation = (y_hat >= y).float()  # Overestimation mask

        # Adapt prepend_value based on the dimensionality of y_hat
        if y_hat.dim() > 1:  # Multi-dimensional
            prepend_value = y_hat[:, 0:1]  # First timestep for each batch in 2D
        else:
            prepend_value = y_hat[0].unsqueeze(0)  # Single value in 1D

        # Calculate absolute differences
        pred_diff = torch.abs(torch.diff(y_hat, dim=-1, prepend=prepend_value))

        # Calculate the quantile threshold, adapting based on the dimensionality
        if y_hat.dim() > 1:
            quantile_threshold = torch.quantile(pred_diff, self.top_quantile, dim=1, keepdim=True)
        else:
            quantile_threshold = torch.tensor([torch.quantile(pred_diff, self.top_quantile)])

        high_penalty_mask = (pred_diff > quantile_threshold).float()

        # Calculate absolute losses with different penalties
        losses = torch.abs(y - y_hat) * (under_estimation * self.penalty_factor + over_estimation)
        losses *= (1 + high_penalty_mask * (self.high_penalty_factor - 1))

        weights = self._compute_weights(y=y, mask=mask)
        weighted_mean = _weighted_mean(losses=losses, weights=weights)
        return weighted_mean


class SelectiveAsymmetricMSELoss(BasePointLoss):
    def __init__(self, horizon_weight=None, penalty_factor=4, high_penalty_factor=8, top_quantile=0.95):
        super(SelectiveAsymmetricMSELoss, self).__init__(horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""])
        self.penalty_factor = penalty_factor
        self.high_penalty_factor = high_penalty_factor
        self.top_quantile = top_quantile

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor, mask: torch.Tensor = None):
        under_estimation = (y_hat < y).float()
        over_estimation = (y_hat >= y).float()

        # Adapt prepend_value based on the dimensionality of y_hat
        if y_hat.dim() > 1:  # Check if y_hat is multidimensional
            prepend_value = y_hat[:, 0:1]  # Taking the first timestep for each batch if it's 2D
        else:
            prepend_value = y_hat[0].unsqueeze(0)  # Ensure it remains 1D

        # Calculate differences
        pred_diff = torch.abs(torch.diff(y_hat, dim=-1, prepend=prepend_value))  # Compute absolute differences

        # Calculate the quantile threshold, adapting based on the dimensionality
        if y_hat.dim() > 1:
            quantile_threshold = torch.quantile(pred_diff, self.top_quantile, dim=1, keepdim=True)  # Each batch separately
        else:
            quantile_threshold = torch.tensor([torch.quantile(pred_diff, self.top_quantile)])  # Global quantile for 1D

        high_penalty_mask = (pred_diff > quantile_threshold).float()

        losses = (y - y_hat) ** 2 * (under_estimation * self.penalty_factor + over_estimation)
        losses *= (1 + high_penalty_mask * (self.high_penalty_factor - 1))

        weights = self._compute_weights(y=y, mask=mask)
        weighted_mean = _weighted_mean(losses=losses, weights=weights)
        return weighted_mean

class MSE(BasePointLoss):
    """Mean Squared Error

    Calculates Mean Squared Error between
    `y` and `y_hat`. MSE measures the relative prediction
    accuracy of a forecasting method by calculating the
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series.

    $$ \mathrm{MSE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^{2} $$

    **Parameters:**<br>
    `horizon_weight`: Tensor of size h, weight for each timestamp of the forecasting window. <br>
    """

    def __init__(self, horizon_weight=None):
        super(MSE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies datapoints to consider in loss.<br>

        **Returns:**<br>
        `mse`: tensor (single value).
        """
        losses = (y - y_hat) ** 2
        weights = self._compute_weights(y=y, mask=mask)
        weighted_mean = _weighted_mean(losses=losses, weights=weights)
        return weighted_mean


class MAE(BasePointLoss):
    """Mean Absolute Error

    Calculates Mean Absolute Error between
    `y` and `y_hat`. MAE measures the relative prediction
    accuracy of a forecasting method by calculating the
    deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    $$ \mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} |y_{\\tau} - \hat{y}_{\\tau}| $$

    **Parameters:**<br>
    `horizon_weight`: Tensor of size h, weight for each timestamp of the forecasting window. <br>
    """

    def __init__(self, horizon_weight=None):
        super(MAE, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies datapoints to consider in loss.<br>

        **Returns:**<br>
        `mae`: tensor (single value).
        """
        losses = torch.abs(y - y_hat)
        weights = self._compute_weights(y=y, mask=mask)
        weighted_mean = _weighted_mean(losses=losses, weights=weights)
        return weighted_mean



if __name__ == '__main__':
    # Example usage
    # Define y_true and y_pred as torch.Tensors, and possibly a mask tensor.
    y_true = torch.tensor([10.0, 20.0, 15.0, 18.0, 20.0])
    y_pred = torch.tensor([9.0, 11.0, 12.0, 21.0, 21.0])

    # mae
    loss_fn = MAE()
    loss_value = loss_fn(y_true, y_pred)
    print(f"MAE Loss: {loss_value}")

    # mse
    loss_fn = MSE()
    loss_value = loss_fn(y_true, y_pred)
    print(f"MSE Loss: {loss_value}")


    loss_fn = SelectiveAsymmetricMAELoss(penalty_factor=1, high_penalty_factor=1)  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MAE Loss: {loss_value}")

    loss_fn = SelectiveAsymmetricMAELoss(penalty_factor=1, high_penalty_factor=1)  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MAE Loss: {loss_value}")

    loss_fn = SelectiveAsymmetricMAELoss(penalty_factor=1, high_penalty_factor=1)  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MAE Loss: {loss_value}")


    # Example SelectiveAsymmetricMSELoss
    loss_fn = SelectiveAsymmetricMSELoss(penalty_factor=1, high_penalty_factor=1)  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MSE Loss: {loss_value}")



    # 测试二维
    y_true = torch.tensor([[10.0, 20.0, 15.0, 18.0, 20.0], [10.0, 20.0, 15.0, 18.0, 20.0]])
    y_pred = torch.tensor([[9.0, 11.0, 12.0, 21.0, 21.0], [9.0, 11.0, 12.0, 21.0, 21.0]])

    # mae
    loss_fn = torch.nn.L1Loss()
    loss_value = loss_fn(y_true, y_pred)
    print(f"MAE Loss: {loss_value}")

    # mse
    loss_fn = torch.nn.MSELoss()
    loss_value = loss_fn(y_true, y_pred)
    print(f"MSE Loss: {loss_value}")

    loss_fn = SelectiveAsymmetricMAELoss(penalty_factor=1, high_penalty_factor=1)  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MAE Loss: {loss_value}")

    loss_fn = SelectiveAsymmetricMSELoss(penalty_factor=1, high_penalty_factor=1)  # Example horizon weight
    loss_value = loss_fn(y_true, y_pred)
    print(f"Selective Asymmetric MSE Loss: {loss_value}")


