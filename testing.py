import datetime
from binance.client import Client as BinanceClient
#import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM

#from sklearn.preprocessing import MinMaxScaler




#bclient = BinanceClient('key', 'secret')


#klines = bclient.get_historical_klines("BNBBTC", bclient.KLINE_INTERVAL_1MINUTE, "1 hour ago UTC")

#print(klines[0])

df = pd.read_csv('BTCRLC.csv')

df = df.drop(["open", "id", "vwp", ], axis=1)

period = 5

counter = 0
other_count = 0
for i in range(int(len(df)/period)):
    b = i * period
    change = df['close'][b:b + period].pct_change().sum()
    if change > 0.006:
        print(change)
        counter = counter + 1
    else:
        other_count = other_count + 1
print(counter)
print(other_count)
print(counter/other_count)
