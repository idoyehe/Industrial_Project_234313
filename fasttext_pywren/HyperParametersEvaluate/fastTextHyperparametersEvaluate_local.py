import fastText as fstTxt
from time import time
import os

files_names = {"dbpedia": ("../Models/dbpedia", "dbpedia.train"),
               "yelp": ("../Models/yelp_review_full", "yelp_review_full.train")}

"""example"""
iter_parameters = \
    [[{'lr': 1, 'lrUpdateRate': 20, 'ws': 6, 'epoch': 12}], [{'lr': 0.6, 'lrUpdateRate': 70, 'ws': 7, 'epoch': 9}],
     [{'lr': 0.5, 'lrUpdateRate': 70, 'ws': 7, 'epoch': 13}], [{'lr': 0.1, 'lrUpdateRate': 100, 'ws': 7, 'epoch': 13}],
     [{'lr': 0.9, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 15}], [{'lr': 0.7, 'lrUpdateRate': 20, 'ws': 4, 'epoch': 13}],
     [{'lr': 0.9, 'lrUpdateRate': 20, 'ws': 6, 'epoch': 8}], [{'lr': 0.3, 'lrUpdateRate': 30, 'ws': 7, 'epoch': 7}],
     [{'lr': 0.2, 'lrUpdateRate': 40, 'ws': 3, 'epoch': 13}], [{'lr': 0.3, 'lrUpdateRate': 40, 'ws': 5, 'epoch': 11}],
     [{'lr': 0.1, 'lrUpdateRate': 20, 'ws': 3, 'epoch': 15}], [{'lr': 0.8, 'lrUpdateRate': 30, 'ws': 7, 'epoch': 6}],
     [{'lr': 0.5, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 10}], [{'lr': 1, 'lrUpdateRate': 100, 'ws': 5, 'epoch': 12}],
     [{'lr': 0.5, 'lrUpdateRate': 90, 'ws': 3, 'epoch': 12}], [{'lr': 0.6, 'lrUpdateRate': 80, 'ws': 3, 'epoch': 9}],
     [{'lr': 0.6, 'lrUpdateRate': 100, 'ws': 3, 'epoch': 15}], [{'lr': 0.2, 'lrUpdateRate': 100, 'ws': 7, 'epoch': 11}],
     [{'lr': 0.3, 'lrUpdateRate': 40, 'ws': 6, 'epoch': 5}], [{'lr': 0.9, 'lrUpdateRate': 20, 'ws': 3, 'epoch': 10}],
     [{'lr': 1, 'lrUpdateRate': 70, 'ws': 3, 'epoch': 13}], [{'lr': 0.2, 'lrUpdateRate': 30, 'ws': 7, 'epoch': 12}],
     [{'lr': 1, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 15}], [{'lr': 1, 'lrUpdateRate': 90, 'ws': 5, 'epoch': 12}],
     [{'lr': 0.7, 'lrUpdateRate': 50, 'ws': 3, 'epoch': 8}], [{'lr': 0.6, 'lrUpdateRate': 30, 'ws': 6, 'epoch': 10}],
     [{'lr': 0.4, 'lrUpdateRate': 40, 'ws': 7, 'epoch': 15}], [{'lr': 0.5, 'lrUpdateRate': 30, 'ws': 3, 'epoch': 11}],
     [{'lr': 0.5, 'lrUpdateRate': 40, 'ws': 7, 'epoch': 15}], [{'lr': 0.5, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 7}],
     [{'lr': 0.5, 'lrUpdateRate': 60, 'ws': 3, 'epoch': 14}], [{'lr': 0.7, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 15}],
     [{'lr': 0.5, 'lrUpdateRate': 90, 'ws': 3, 'epoch': 13}], [{'lr': 0.4, 'lrUpdateRate': 80, 'ws': 6, 'epoch': 6}],
     [{'lr': 0.7, 'lrUpdateRate': 70, 'ws': 4, 'epoch': 13}], [{'lr': 0.6, 'lrUpdateRate': 40, 'ws': 6, 'epoch': 12}],
     [{'lr': 1, 'lrUpdateRate': 70, 'ws': 3, 'epoch': 14}], [{'lr': 0.2, 'lrUpdateRate': 50, 'ws': 6, 'epoch': 5}],
     [{'lr': 1, 'lrUpdateRate': 90, 'ws': 7, 'epoch': 13}], [{'lr': 1, 'lrUpdateRate': 70, 'ws': 3, 'epoch': 7}],
     [{'lr': 0.2, 'lrUpdateRate': 20, 'ws': 7, 'epoch': 9}], [{'lr': 0.3, 'lrUpdateRate': 100, 'ws': 5, 'epoch': 7}],
     [{'lr': 0.8, 'lrUpdateRate': 90, 'ws': 7, 'epoch': 10}], [{'lr': 0.2, 'lrUpdateRate': 20, 'ws': 4, 'epoch': 11}],
     [{'lr': 1, 'lrUpdateRate': 50, 'ws': 7, 'epoch': 5}], [{'lr': 0.5, 'lrUpdateRate': 40, 'ws': 4, 'epoch': 9}],
     [{'lr': 0.9, 'lrUpdateRate': 80, 'ws': 7, 'epoch': 9}], [{'lr': 0.4, 'lrUpdateRate': 40, 'ws': 4, 'epoch': 14}],
     [{'lr': 0.2, 'lrUpdateRate': 60, 'ws': 4, 'epoch': 15}], [{'lr': 0.9, 'lrUpdateRate': 40, 'ws': 7, 'epoch': 14}]]

K = 3

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
        current_params = current_params[0]
        results = list()
        for index in range(K):
            results.append(
                map_k_fold_cross_validation(files_names["dbpedia"][0], files_names["dbpedia"][1], current_params, K, index))
        total_results["Results"].append(reducer_average_validator(results))
    total_results["Duration"] = time() - start

    print(total_results)
