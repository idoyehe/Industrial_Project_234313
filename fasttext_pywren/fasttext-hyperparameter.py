import fastText as fstTxt
import pywren_ibm_cloud as pywren
from time import time
from ExecuterWrapper.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN
GIVEN_PARAMS = {
    "lr": 0.1,
    "dim": 100,
    "ws": 5,
    "epoch": 5,
    "minCount": 1}

files_names = {"dbpedia": "dbpedia.train",
               "yelp": "yelp_review_full.train"}


chosen_model = files_names["yelp"]

bucketname = 'fasttext-train-bucket'
train_file = [bucketname + "/" + files_names["dbpedia"]]


def map_k_cross_validation(valid_index, stream):
    path_docker = "/fasttext/validation/"
    source_path = path_docker + chosen_model
    valid_path = path_docker + "source.vaild"
    train_path = path_docker + "source.train"

    input_file = stream
    train_file = open(train_path, 'w')
    valid_file = open(valid_path, 'w')

    current_index = 0
    for line in input_file:
        str_line = str(line) + "\n"
        if current_index % 4 == valid_index:
            valid_file.write(str_line)
        else:
            train_file.write(str_line)

        current_index += 1

    valid_file.close()
    train_file.close()

    parameters_dict = {
        "lr": 0.1,
        "dim": 100,
        "ws": 5,
        "epoch": 5,
        "minCount": 1}

    to_valid_model = fstTxt.train_supervised(train_path, **parameters_dict)
    result = to_valid_model.test(valid_path)
    return {"precision": result[1], "recall": result[2]}




def reducer_average_validator(results):
    avg_precision = 0
    avg_recall = 0
    for current in results:
        avg_precision += current["precision"]
        avg_recall += current["recall"]

    avg_precision /= len(results)
    avg_recall /= len(results)
    avg_result = {"precision": avg_precision, "recall": avg_recall}
    return avg_result


def map_evaluate_parameters(key, data_stream):
    K = 4

    stream = [line.decode("utf-8") for line in data_stream.read().splitlines()]  # bottleneck

    k_list = [(i, stream) for i in range(K)]  # bottleneck
    subpw = pywren.ibm_cf_executor(runtime="fasttext-hyperparameter")
    subpw.map_reduce(map_k_cross_validation, k_list, reducer_average_validator, reducer_wait_local=False)
    result_object = subpw.get_result()
    return result_object


start = time()
pw = pywren.ibm_cf_executor(runtime="fasttext-hyperparameter")
pw.map(map_evaluate_parameters, train_file)
res = pw.get_result()
end = time()
print("\nDuration: " + str(end - start) + " Sec")
print("Results: ", res)
