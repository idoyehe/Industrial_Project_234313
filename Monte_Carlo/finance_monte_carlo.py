from numpy import exp, random, ones_like
import matplotlib.pyplot as plt
from scipy.stats import norm
from ExecuterWrapper.executerWrapper import ExecuterWrap, Location

exe_location = Location.LOCAL


class StockData:
    total_forecasts = 100
    days2predict = 1095

    def __init__(self, name, drift, std_dev, last_value):
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


gold = StockData(name="GOLD", drift=0.000142559, std_dev=0.010561899, last_value=1296.5)
mlnx = StockData(name="Mellanox", drift=0.000581742829, std_dev=0.029879238, last_value=64.7)
ibm = StockData(name="IBM", drift=0.000091967236, std_dev=0.012404562, last_value=153.42)
nvda = StockData(name="Nvdia", drift=0.000936809, std_dev=0.027145343, last_value=193.5)

current_stock = gold
print("Current Stock: " + current_stock.name)
print("Total Forecasts: " + str(StockData.total_forecasts))
print("Days to Predict: " + str(current_stock.days2predict))

iterdata = [[]] * StockData.total_forecasts


def my_map_function(curr=None):
    return current_stock.forecast()


def my_reduce_function(list_of_lists):
    end = current_stock.days2predict
    hist_end = [frc[end] for frc in list_of_lists]
    mid = int(current_stock.days2predict / 2)
    hist_mid = [frc[mid] for frc in list_of_lists]
    min_forecast = []
    max_forecast = []
    for frc in list_of_lists:
        if len(min_forecast) == 0 or (frc[end] < min_forecast[end]):  # setting worst case by minimum last day
            min_forecast = frc

        if len(max_forecast) == 0 or (frc[end] > max_forecast[end]):  # setting best case by maximum last day
            max_forecast = frc
    return {"min": min_forecast, "max": max_forecast, "hist_mid": hist_mid, "hist_end": hist_end}


executer = ExecuterWrap()
executer.set_location(exe_location)
result_obj = executer.map_reduce_execution(my_map_function, iterdata, my_reduce_function)

print("Stock values minimum forecast: ")
print(result_obj["min"])

print("Stock values maximum forecast: ")
print(result_obj["max"])

'''Minimum forecast plot'''
plt.plot([x for x in range(current_stock.days2predict + 1)], result_obj["min"])
plt.grid(True)
plt.xlabel("Days")
plt.ylabel("Value [$]")
plt.title("Minimum Forecast")
plt.show()

'''Maximum forecast plot'''
plt.plot([x for x in range(current_stock.days2predict + 1)], result_obj["max"])
plt.grid(True)
plt.title("Maximum Forecast")
plt.xlabel("Days")
plt.ylabel("Value [$]")
plt.show()

'''Histogram for mid prediction forecast plot'''
plt.hist(result_obj["hist_mid"])
plt.grid(True)
plt.title("Mid prediction period histogram")
plt.ylabel("Count")
plt.xlabel("Value [$]")
plt.show()

'''Histogram for end prediction forecast plot'''
plt.hist(result_obj["hist_end"])
plt.grid(True)
plt.title("End prediction period histogram")
plt.ylabel("Count")
plt.xlabel("Value [$]")
plt.show()
