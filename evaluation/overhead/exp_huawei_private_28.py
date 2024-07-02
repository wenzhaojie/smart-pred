from exp_with_one_trace import ExpOneTrace






def start_exp():
    my_exp = ExpOneTrace(
        dataset_name="HuaweiPrivateDataset", function_name="28", start_day=0, end_day=4,
    )

    my_exp.init_baselines(
        # baselines_name_list=["Keepalive", "Crane", "Icebreaker", "SmartPred", "NHITS", "NBEATS", "PatchTST", "DLinear"],
        baselines_name_list=["Maxvalue", "Minvalue", "Avgvalue","Movingavg", "Movingmax", "Movingmin", "Quantile", "SmartPred", "NHITS", "NBEATS", "DLinear", "Icebreaker", "Crane"],
        baselines_config_dict={
            "NHITS": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "NBEATS": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "PatchTST": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "DLinear": {
                "seq_len": 1440,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "period": 1440,
                "max_steps": 100,
                "amplify": 1.0,
                "loss": "MSELoss",
            },
            "Movingavg": {
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "moving_window": 40,
                "is_scaler": False,
                "is_round": False,
                "max_steps": 100,
                "amplify": 1.0,
            },
            "Movingmax": {
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "moving_window": 20,
                "is_scaler": False,
                "is_round": False,
                "max_steps": 100,
                "amplify": 1.0,
            },
            "Movingmin": {
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "moving_window": 20,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.0,
            },
            "Maxvalue": {
                "seq_len": 1440 * 3,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.0,
            },
            "Minvalue": {
                "seq_len": 1440 * 3,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.0,
            },
            "Avgvalue": {
                "seq_len": 1440 * 3,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.0,
            },
            "OpenFaaS": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "amplify": 1.0,
                "moving_window": 10,
            },
            "Crane": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 3,
                "pred_len": 1440,
                "amplify": 1.0
            },
            "Quantile": {
                "seq_len": 1440 * 3,
                "pred_len": 1440,
                "is_scaler": False,
                "is_round": False,
                "quantile": 0.5,
                "period": 1440,
                "amplify": 1.0
            },
            "Icebreaker": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 5,
                "pred_len": 1440,
                "amplify": 1.0
            },
            "Keepalive": {
                "is_scaler": False,
                "is_round": False,
                "seq_len": 1440 * 3,
                "pred_len": 1,
                "moving_window": 10,
                "amplify": 1.0,
            },
            "SmartPred": {
                "seq_len": 1440 * 4,
                "pred_len": 1440,
                "period_length": 1440,
                "complex_model_list": ["NHITS", "NBEATS", "PatchTST", "DLinear"],
                "simple_model_list": ["Movingmax_model", "Movingavg_model", "Maxvalue_model", "Histogram_model"],
                "loss": "SelectiveAsymmetricMAELoss",
                "determine_ratio": 0.95,
                "is_scaler": False,
                "is_round": False,
                "amplify": 1.03,
                "is_complex": False,
                "delta_t": 120,
                "epsilon": 1.5,
                "bursty_quantile": 0.99,
            },
        }
    )

    my_exp.evaluation()


if __name__ == "__main__":
    start_exp()