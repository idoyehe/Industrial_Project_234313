import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scpy
from ExecuterWrapper.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN
MAP_INSTANCES = 1000


class StockData:
    forecasts_per_map = 100
    days2predict = 1095

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
ibm_2016_2017_2018 = StockData(title="IBM Based 2016, 2017, 2018 (end of October)", drift=-0.000295076840817479, std_dev=0.0118636061422374,
                  last_value=116.49)
ibm_2014_2015_2016 = StockData(title="IBM 2014, 2015, 2016", drift=-0.00022513546014255100, std_dev=0.0121678341323272,
                               last_value=166.44)

intel_10 = StockData(title="Intel Based last 10 years", drift=0.000359910620036371, std_dev=0.0155512505615464, last_value=48.29)
intel_2016_2017_2018 = StockData(title="Intel Based 2016, 2017, 2018 (end of October)", drift=0.00038623298498201, std_dev=0.0148876589619472, last_value=48.29)
intel_2014_2015_2016 = StockData(title="Intel 2014, 2015, 2016", drift=0.00036084284127726200, std_dev=0.0143579745383959,
                                 last_value=36.79)

current_stock = ibm_10

print("Current Stock: " + current_stock.title)
print("Total Forecasts: " + str(MAP_INSTANCES * StockData.forecasts_per_map))
print("Days to Predict: " + str(current_stock.days2predict))

iterate_data = [()] * MAP_INSTANCES


def map_function(data=None):
    end = current_stock.days2predict
    mid = int(end / 2)
    hist_end = list()
    hist_mid = list()
    min_f = None
    max_f = None
    for i in range(current_stock.forecasts_per_map):
        frc = current_stock.single_forecast_generator()
        hist_end.append(frc[end])
        hist_mid.append(frc[mid])
        if min_f is None or (frc[end] < min_f[end]):  # setting worst case by minimum last day
            min_f = frc
        if max_f is None or (frc[end] > max_f[end]):  # setting best case by maximum last day
            max_f = frc
    return min_f, max_f, hist_mid, hist_end


def reduce_function(results, futures):
    print(np.__version__)  # in order to import numpy
    end = current_stock.days2predict
    hist_end = list()
    hist_mid = list()
    min_f = None
    max_f = None
    for single_map_result in results:
        hist_end.extend(single_map_result[3])
        hist_mid.extend(single_map_result[2])
        if min_f is None or (single_map_result[0][end] < min_f[end]):  # setting worst case by minimum last day
            min_f = single_map_result[0]
        if max_f is None or (single_map_result[1][end] > max_f[end]):  # setting best case by maximum last day
            max_f = single_map_result[1]

    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]
    return {"run_statuses": run_statuses, "invoke_statuses": invoke_statuses, "results": (min_f, max_f, hist_mid, hist_end)}


executor = ExecutorWrap(MAP_INSTANCES, "finance_monte_carlo_" + str(MAP_INSTANCES * StockData.forecasts_per_map))
executor.set_location(exe_location)
result_obj = executor.map_reduce_execution(map_function, iterate_data, reduce_function)

# print("Stock values minimum forecast: ")
# print(result_obj[0])
#
# print("Stock values maximum forecast: ")
# print(result_obj[1])
#
# '''Minimum forecast plot'''
# min_forecast = result_obj[0]
# plt.plot([x for x in range(current_stock.days2predict + 1)], min_forecast)
# plt.grid(True)
# plt.xlabel("Days")
# plt.ylabel("Value [$]")
# plt.title("Minimum Forecast")
# plt.xticks(np.arange(0, StockData.days2predict + 1, 150))
# plt.show()
#
# '''Maximum forecast plot'''
# max_forecast = result_obj[1]
# plt.plot([x for x in range(current_stock.days2predict + 1)], max_forecast)
# plt.grid(True)
# plt.title("Maximum Forecast")
# plt.xlabel("Days")
# plt.ylabel("Value [$]")
# plt.xticks(np.arange(0, StockData.days2predict + 1, 150))
# plt.show()
#
# '''Histogram for mid prediction forecast plot'''
# mid_data = result_obj[2]
# plt.hist(mid_data, bins='auto')
# plt.grid(True)
# plt.title("Mid prediction period histogram")
# plt.ylabel("# Forecasts")
# plt.xlabel("Value [$]")
# plt.show()
#
# '''Histogram for end prediction forecast plot'''
# end_data = result_obj[3]
# plt.hist(end_data, bins='auto')
# plt.grid(True)
# plt.title("End prediction period histogram")
# plt.ylabel("# Forecasts")
# plt.xlabel("Value [$]")
# plt.show()
