import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scpy
from ExecuterWrapper.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN
MAP_INSTANCES = 1000


class StockData:
    forecasts_per_map = 100
    days2predict = 365

    def __init__(self, title, drift, std_dev, last_value):
        self.title = title
        self.last_value = last_value
        self.std_dev = std_dev
        self.drift = drift

    def single_forecast_generator(self):
        predicts_est = [self.last_value]
        for predict in range(1, self.days2predict + 1):
            rand = np.random.rand()
            pow_r = scpy.norm.ppf(rand)
            predicts_est.append(predicts_est[predict - 1] * np.exp(self.drift + (self.std_dev * pow_r)))
        return predicts_est


ibm_10 = StockData(title="IBM Based last 10 years", drift=0.0000579602177315899, std_dev=0.0119319087951656, last_value=116.49)
ibm_3 = StockData(title="IBM Based last 3 years", drift=-0.000418352004242025, std_dev=0.0120446535109423, last_value=116.49)
ibm_2014_2015_2016 = StockData(title="IBM 2014, 2015, 2016", drift=-0.00022513546014255100, std_dev=0.0121678341323272, last_value=166.44)

intel_10 = StockData(title="Intel Based last 10 years", drift=0.000359910620036371, std_dev=0.0155512505615464, last_value=48.29)
intel_3 = StockData(title="Intel Based last 3 years", drift=0.000169007164760754, std_dev=0.0152474310285415, last_value=48.29)
intel_2014_2015_2016 = StockData(title="Intel 2014, 2015, 2016", drift=0.00036084284127726200, std_dev=0.0143579745383959, last_value=36.79)

current_stock = intel_3

print("Current Stock: " + current_stock.title)
print("Total Forecasts: " + str(MAP_INSTANCES * StockData.forecasts_per_map))
print("Days to Predict: " + str(current_stock.days2predict))

iterdata = [()] * MAP_INSTANCES


def map_function(data=None):
    end = current_stock.days2predict
    mid = int(end / 2)
    hist_end = list()
    hist_mid = list()
    for i in range(StockData.forecasts_per_map):
        frc = current_stock.single_forecast_generator()
        hist_end.append(frc[end])
        hist_mid.append(frc[mid])
    return hist_mid, hist_end


def reduce_function(results):
    print(np.__version__)# in order to import numpy
    hist_end = list()
    hist_mid = list()
    for single_map_result in results:
        hist_end.extend(single_map_result[1])
        hist_mid.extend(single_map_result[0])
    return {"futures": None, "results": (hist_mid, hist_end)}


executor = ExecutorWrap(MAP_INSTANCES)
executor.set_location(exe_location)
result_obj = executor.map_reduce_execution(map_function, iterdata, reduce_function)

'''Histogram for mid prediction forecast plot'''
mid_data = result_obj[0]
print("MID Histogram:")
print(mid_data)
plt.hist(mid_data, bins='auto')
plt.grid(True)
plt.title("Mid prediction period histogram")
plt.ylabel("Count")
plt.xlabel("Value [$]")
plt.show()

'''Histogram for end prediction forecast plot'''
end_data = result_obj[1]
print("END Histogram:")
print(end_data)
plt.hist(end_data, bins='auto')
plt.grid(True)
plt.title("End prediction period histogram")
plt.ylabel("Count")
plt.xlabel("Value [$]")
plt.show()
