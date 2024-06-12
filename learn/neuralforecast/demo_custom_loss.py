from neuralforecast.utils import AirPassengersDF
from smart_pred.utils.nf_loss import SelectiveAsymmetricMAELoss


Y_df = AirPassengersDF # Defined in neuralforecast.utils
Y_df.head()

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN

horizon = 12

# Try different hyperparmeters to improve accuracy.
models = [LSTM(h=horizon,                    # Forecast horizon
               max_steps=500,                # Number of steps to train
               scaler_type='standard',       # Type of scaler to normalize data
               encoder_hidden_size=64,       # Defines the size of the hidden state of the LSTM
               decoder_hidden_size=64,
               loss=SelectiveAsymmetricMAELoss()
               ),     # Defines the number of hidden units of each layer of the MLP decoder
          NHITS(h=horizon,                   # Forecast horizon
                input_size=2 * horizon,      # Length of input sequence
                max_steps=100,               # Number of steps to train
                n_freq_downsample=[2, 1, 1],
                loss=SelectiveAsymmetricMAELoss()
                ) # Downsampling factors for each stack output
          ]
nf = NeuralForecast(models=models, freq='M')
nf.fit(df=Y_df)

Y_hat_df = nf.predict()

Y_hat_df = Y_hat_df.reset_index()
Y_hat_df.head()

import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (20, 7))
plot_df = pd.concat([Y_df, Y_hat_df]).set_index('ds') # Concatenate the train and forecast dataframes
plot_df[['y', 'LSTM', 'NHITS']].plot(ax=ax, linewidth=2)

ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()

plt.show()


