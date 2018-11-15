import pywren_ibm_cloud as pywren
from time import time
from numpy import exp, random
import matplotlib.pyplot as plt
from scipy.stats import norm


class StockData:
    def __init__(self, name, drift, std_dev, last_value, days2predict):
        self.name = name
        self.days2predict = days2predict
        self.last_value = last_value
        self.std_dev = std_dev
        self.drift = drift
        self.forcast_length = days2predict + 1

    def prediction(self):
        predicts_est = [self.last_value]
        for predict in range(1, self.forcast_length):
            rand = random.rand()
            pow_r = norm.ppf(rand)
            predicts_est.append(predicts_est[predict - 1] * exp(self.drift + (self.std_dev * pow_r)))
        return predicts_est


ACTIONS = 1500
TOTAL = ACTIONS
print("Total Prediction: " + str(TOTAL))

gold = StockData(name="GOLD", drift=0.000142559, std_dev=0.010561899, last_value=1296.5, days2predict=1095)
mlnx = StockData(name="Mellanox", drift=0.000581742829, std_dev=0.029879238, last_value=64.7, days2predict=1095)
ibm = StockData(name="IBM", drift=0.000091967236, std_dev=0.012404562, last_value=153.42, days2predict=1095)
nvda = StockData(name="Nvdia", drift=0.000936809, std_dev=0.027145343, last_value=193.5, days2predict=1095)

stock = ibm
print("Current Stock: " + stock.name)

iterdata = [[]] * ACTIONS


def my_map_function():
    return stock.prediction()


def my_reduce_function(list_of_lists):
    end = stock.days2predict
    hist_end = [frc[end] for frc in list_of_lists]
    mid = int(stock.days2predict / 2)
    hist_mid = [frc[mid] for frc in list_of_lists]
    min_forecast = []
    max_forecast = []
    for frc in list_of_lists:
        if len(min_forecast) == 0 or (frc[stock.days2predict] < min_forecast[stock.days2predict]):
            min_forecast = frc

        if len(max_forecast) == 0 or (frc[stock.days2predict] > max_forecast[stock.days2predict]):
            max_forecast = frc
    return {"min": min_forecast, "max": max_forecast, "hist_mid": hist_mid, "hist_end": hist_end}


"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""

# FLAG = "LOCAL"
FLAG = "CLOUD"
if FLAG == "LOCAL":
    start_time = time()
    for i in range(ACTIONS):
        iterdata[i] = my_map_function()
    result_list = my_reduce_function(iterdata)
else:
    start_time = time()
    pw = pywren.ibm_cf_executor()
    pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
    result_list = pw.get_result()

elapsed = time()
print("\nDuration: " + str(elapsed - start_time) + " Sec")

print("Stock values minimum forecast: ")
print(result_list["min"])

print("Stock values maximum forecast: ")
print(result_list["max"])

plt.plot([x for x in range(stock.days2predict + 1)], result_list["min"])
plt.title("Minimum Forecast")
plt.show()
plt.plot([x for x in range(stock.days2predict + 1)], result_list["max"])
plt.title("Maximum Forecast")
plt.show()
plt.hist(result_list["hist_mid"], bins='auto')
plt.title("Mid period histogram")
plt.show()
plt.hist(result_list["hist_end"], bins='auto')
plt.title("End period histogram")
plt.show()
