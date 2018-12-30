import fastText as fstTxt
import pywren_ibm_cloud as pywren
from time import time

files_names = {"dbpedia": "dbpedia.train",
               "yelp": "yelp_review_full.train"}

bucketname = 'fasttext-train-bucket'

files_to_predict = dict()

for key, value in files_names.items():
    files_names[key] = bucketname + '/' + value

iter_parameters = [{
    "lr": 0.3,
    "dim": 100,
    "ws": 5,
    "epoch": 7,
    "minCount": 1},
    {"lr": 0.1,
     "dim": 100,
     "ws": 5,
     "epoch": 5,
     "minCount": 1}]


class HyperParameterTuning(object):
    def __init__(self, path_docker: str, parameters_dict: dict, k_value: int, index: int):
        self.path_docker = path_docker
        self.parameters_dict = parameters_dict
        self.k_value = k_value
        self.index = index

    def map_k_cross_validation(self, data_stream):
        valid_path = self.path_docker + "source.valid"
        train_path = self.path_docker + "source.train"

        valid_index = 0  # should get as function argument

        train_file = open(train_path, 'w')
        valid_file = open(valid_path, 'w')

        current_index = 0
        for line in data_stream.splitlines():
            str_line = line  # line.decode('utf-8') incase data comes from COS
            str_line += "\n"
            if current_index % self.k_value == valid_index:
                valid_file.write(str_line)
            else:
                train_file.write(str_line)

            current_index += 1

        valid_file.close()
        train_file.close()

        to_valid_model = fstTxt.train_supervised(train_path, **self.parameters_dict)
        result = to_valid_model.test(valid_path)
        return {"pre": result[1], "rec": result[2]}


def map_evaluate_parameters(lr, dim, ws, epoch, minCount):
    k_value = 4

    k_list = [i for i in range(k_value)]  # bottleneck

    parameters = {"lr": lr, "dim": dim, "ws": ws, "epoch": epoch, "minCount": minCount}

    def k_cross_validtion_wrap(index):
        data_stream = open("/fasttext/validation/dbpedia.train", "r").read()  # should be with COS
        hyper = HyperParameterTuning("/fasttext/validation/", parameters, k_value, index)
        return hyper.map_k_cross_validation(data_stream)

    def reducer_average_validator(results):
        avg_precision = 0
        avg_recall = 0
        for res in results:
            avg_precision += res["pre"]
            avg_recall += res["rec"]

        avg_precision /= len(results)
        avg_recall /= len(results)
        return {"precision": avg_precision, "recall": avg_recall}

    sub_pw = pywren.ibm_cf_executor(runtime="fasttext-hyperparameter")
    sub_pw.map_reduce(k_cross_validtion_wrap, k_list, reducer_average_validator, reducer_wait_local=False)
    return sub_pw.get_result()


start = time()
pw = pywren.ibm_cf_executor(runtime="fasttext-hyperparameter")
pw.map(map_evaluate_parameters, iter_parameters)
pw_res = pw.get_result()
end = time()
print("\nDuration: " + str(end - start) + " Sec")
print("Results: ", pw_res)
