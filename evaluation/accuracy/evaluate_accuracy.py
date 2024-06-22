from smart_pred.model.local.histogram import Histogram_model, OpenFaaS_model
from smart_pred.model.local.fourier import Crane_dsp_model, Icebreaker_model


model_dict = {
    "OpenFaaS": OpenFaaS_model(),
    "Histogram": OpenFaaS_model(),
    "Crane": Crane_dsp_model(),
    "Icebreaker": Icebreaker_model(),
    "Histogram_model": Histogram_model(),
}

model_config_dict = {
    "OpenFaaS": {
        "is_scaler": True,
        "is_round": False,
        "seq_len": 1440 * 3,
        "pred_len": 1440,
    },
    "NHITS": {
        "seq_len": 1440,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
        "max_steps": 100,
    },
    "NBEATS": {
        "seq_len": 1440,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
        "max_steps": 100,
    },
    "PatchTST": {
        "seq_len": 1440,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
        "max_steps": 100,
    },
    "DLinear": {
        "seq_len": 1440,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "period": 1440,
        "max_steps": 100,
    },
    "Movingavg_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 20,
        "is_scaler": False,
        "is_round": True,
        "max_steps": 100,
    },
    "Movingmax_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 20,
        "is_scaler": False,
        "is_round": True,
        "max_steps": 100,
    },
    "Movingmin_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1,
        "moving_window": 20,
        "is_scaler": False,
        "is_round": True,
    },
    "Maxvalue_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    },
    "Minvalue_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    },
    "Avgvalue_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    },
    "Quantile_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
        "quantile": 0.5,
    },
    "Histogram_model": {
        "seq_len": 1440 * 3,
        "pred_len": 1440,
        "is_scaler": False,
        "is_round": False,
    },
}

trace_dict = {
    "public": {
        "selected_trace": [3, 4, 9, 10, 11, 15, 21, 26, ],
    },
    "private": {
        "selected_trace":  [28, 39, 40, 60, 72, 73, 75, 86, 89, 96, 100, 127]
    },
    "azure": {
        "selected_trace": [
            "1ad3a3335bb2c127474fbfd71ef278319ad4c58ed0bfe4d5e152f77d18ebbe33",
            "1b0736fd9c51899497cfa21385cabab7dd9c37f64af4234eed16c4c48edf8b64",
            "3d35c606b3178f64d36dc552a9a16b4c8bfd39cc67e654d55b957c1f28f5b7f7",
            "4a83a242c6293aa5690e6d844c91bc2d2c8a4e74e18aa1c1c30be726480035ec",
            "7d3d5a1ef74ec2cdd055c444d1f735682f3436c8be39e43ba6952af1bcba1ee1",
        ],
    },
}

def get_metric_from_trace(train_data, test_data):



    pass
