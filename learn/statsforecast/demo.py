from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

import os
os.environ['NIXTLA_ID_AS_COL'] = '1'

df = AirPassengersDF

print(df)
sf = StatsForecast(
    models = [AutoARIMA(season_length = 12)],
    freq = 'ME'
)

sf.fit(df)
# res = sf.predict(h=12, level=[95])
res = sf.predict(h=12,)

print(res["AutoARIMA"].values)