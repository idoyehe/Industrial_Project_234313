import pywren_ibm_cloud as pywren
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time

stocks = ['ibm', 'gold', 'nvda', 'mlnx', 'mrvl', 'msft', 'goog', 'fb', 'amzn']

days2predict = 1095


def predict_function(stock_sym):
    data = pd.DataFrame()
    data[stock_sym] = wb.DataReader(stock_sym, data_source='yahoo', start='2007-1-1')['Adj Close']
    log_returns = np.log(1 + data.pct_change())
    log_returns.tail()
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    iterations = 20
    daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(days2predict, iterations)))
    stock_price_list = np.zeros_like(daily_returns)
    stock_price_list[0] = data.iloc[-1]
    for t in range(1, days2predict):
        stock_price_list[t] = stock_price_list[t - 1] * daily_returns[t]
    return stock_price_list


FLAG = "LOCAL"
# FLAG = "CLOUD"
if FLAG == "LOCAL":
    start_time = time()
    for i in range(len(stocks)):
        stocks[i] = predict_function(stocks[i])
    result_list = stocks
else:
    start_time = time()
    pw = pywren.ibm_cf_executor()
    pw.map(predict_function, stocks,)
    result_list = pw.get_result()

elapsed = time()

print("Stock values prediction: ")
print(result_list)

print("\nDuration: " + str(elapsed - start_time) + " Sec")

for price_list in result_list:
    plt.figure(figsize=(20, 10))
    plt.plot(price_list)
    plt.show()
