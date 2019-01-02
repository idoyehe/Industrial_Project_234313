import fastText as fstTxt
from time import time
import os

files_names = {"dbpedia": ("../Models/dbpedia", "dbpedia.train"),
               "yelp": ("../Models/yelp_review_full", "yelp_review_full.train")}

"""example"""
iter_parameters = \
    [{'lr': 0.3, 'lrUpdateRate': 80, 'ws': 6, 'epoch': 5}]

K = 1

def fastText_evaluate(parameters_dict, train_path, test_path):
    to_valid_model = fstTxt.train_supervised(train_path, **parameters_dict)
    result = to_valid_model.test(test_path)
    return {"precision": result[1], "recall": result[2]}


def map_k_fold_cross_validation(source_path, model_name, params_dict, k_value, valid_index):
    valid_path = source_path + "/source.valid"
    train_path = source_path + "/source.train"

    source_file = open(source_path + "/" + model_name, 'r')
    train_file = open(train_path, 'w')
    valid_file = open(valid_path, 'w')

    current_index = 0
    for line in source_file.read().splitlines():
        str_line = line + "\n"
        if current_index % k_value == valid_index:
            valid_file.write(str_line)
        else:
            train_file.write(str_line)

        current_index += 1

    valid_file.close()
    train_file.close()
    source_file.close()

    res = fastText_evaluate(params_dict, train_path, valid_path)

    os.remove(valid_path)
    os.remove(train_path)
    return res


def reducer_average_validator(results):
    avg_precision = 0
    avg_recall = 0
    for res in results:
        avg_precision += res["precision"]
        avg_recall += res["recall"]

    avg_precision /= len(results)
    avg_recall /= len(results)
    return {"precision": avg_precision, "recall": avg_recall}


total_results = {"Results": list(), "Duration": None}

for i in range(3):
    start = time()
    for current_params in iter_parameters:
        results = list()
        for index in range(K):
            results.append(
                map_k_fold_cross_validation(files_names["yelp"][0], files_names["yelp"][1], current_params, K, index))
        total_results["Results"].append(reducer_average_validator(results))
    total_results["Duration"] = time() - start

    print(total_results)
