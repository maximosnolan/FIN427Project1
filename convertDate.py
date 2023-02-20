import pandas as pd
from datetime import datetime
data = pd.read_csv("dailyPrices.csv")

data['DATE'] = data['date'].str[0:2] + '-' + data['date'].str[2:5] + '-' + data['date'].str[7:9]
data['Month'] = datetime.strptime(data['date'].str[2:5].str(), '%b').month
data.to_csv("dailyPrices.csv")
