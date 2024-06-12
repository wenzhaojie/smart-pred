from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.utils import AirPassengersDF

nf = NeuralForecast(
    models = [TimeLLM(
        input_size=24,
        h=12,
        max_steps=100
    )],
    freq = 'M'
)

nf.fit(df=AirPassengersDF)
nf.predict()