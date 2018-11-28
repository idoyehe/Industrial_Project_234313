from numpy import exp, random, arange
import matplotlib.pyplot as plt
from scipy.stats import norm
from ExecuterWrapper.executerWrapper import ExecuterWrap, Location

exe_location = Location.CLOUD


class StockData:
    total_forecasts = 1200
    days2predict = 1095

    def __init__(self, name, drift, std_dev, last_value, ticks):
        self.ticks = ticks
        self.name = name
        self.last_value = last_value
        self.std_dev = std_dev
        self.drift = drift

    def forecast(self):
        predicts_est = [self.last_value]
        for predict in range(1, self.days2predict + 1):
            rand = random.rand()
            pow_r = norm.ppf(rand)
            predicts_est.append(predicts_est[predict - 1] * exp(self.drift + (self.std_dev * pow_r)))
        return predicts_est


gold = StockData(name="GOLD", drift=0.000142559, std_dev=0.010561899, last_value=1296.5, ticks=400)
mlnx = StockData(name="Mellanox", drift=0.000581742829, std_dev=0.029879238, last_value=64.7, ticks=10)
ibm = StockData(name="IBM", drift=0.000091967236, std_dev=0.012404562, last_value=153.42, ticks=50)
nvda = StockData(name="Nvdia", drift=0.000936809, std_dev=0.027145343, last_value=193.5, ticks=50)

current_stock = gold
print("Current Stock: " + current_stock.name)
print("Total Forecasts: " + str(StockData.total_forecasts))
print("Days to Predict: " + str(current_stock.days2predict))

iterdata = [[]] * StockData.total_forecasts


def my_map_function(curr=None):
    return current_stock.forecast()


def my_reduce_function(results):
    end = current_stock.days2predict
    hist_end = [frc[end] for frc in results]
    mid = int(current_stock.days2predict / 2)
    hist_mid = [frc[mid] for frc in results]
    min_forecast = []
    max_forecast = []
    for frc in results:
        if len(min_forecast) == 0 or (frc[end] < min_forecast[end]):  # setting worst case by minimum last day
            min_forecast = frc

        if len(max_forecast) == 0 or (frc[end] > max_forecast[end]):  # setting best case by maximum last day
            max_forecast = frc
    return {"futures": "",
            "result_obj": {"min": min_forecast, "max": max_forecast, "hist_mid": hist_mid, "hist_end": hist_end}}


executer = ExecuterWrap()
executer.set_location(exe_location)
result_obj = executer.map_reduce_execution(my_map_function, iterdata, my_reduce_function)

print("Stock values minimum forecast: ")
print(result_obj["min"])

print("Stock values maximum forecast: ")
print(result_obj["max"])

'''Minimum forecast plot'''
min_forecast = result_obj["min"]
plt.plot([x for x in range(current_stock.days2predict + 1)], min_forecast)
plt.grid(True)
plt.xlabel("Days")
plt.ylabel("Value [$]")
plt.title("Minimum Forecast")
plt.xticks(arange(0, StockData.days2predict + 1, 150))
plt.show()

'''Maximum forecast plot'''
max_forecast = result_obj["max"]
plt.plot([x for x in range(current_stock.days2predict + 1)], max_forecast)
plt.grid(True)
plt.title("Maximum Forecast")
plt.xlabel("Days")
plt.ylabel("Value [$]")
plt.xticks(arange(0, StockData.days2predict + 1, 150))
plt.show()

'''Histogram for mid prediction forecast plot'''
mid_data = result_obj["hist_mid"]
plt.hist(mid_data, bins='auto')
plt.grid(True)
plt.title("Mid prediction period histogram")
plt.ylabel("Count")
plt.xlabel("Value [$]")
plt.xticks(arange(min(mid_data), max(mid_data), current_stock.ticks))
plt.show()

'''Histogram for end prediction forecast plot'''
end_data = result_obj["hist_end"]
plt.hist(end_data, bins='auto')
plt.grid(True)
plt.title("End prediction period histogram")
plt.ylabel("Count")
plt.xlabel("Value [$]")
plt.xticks(arange(min(end_data), max(end_data), current_stock.ticks))
plt.show()
