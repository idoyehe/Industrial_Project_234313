import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate.localGenericHyperparameterEvaluate import *
from fasttext_pywren.HyperParametersEvaluate.randomHyperparameters import random_search
import csv


# from fasttext_pywren.HyperParametersEvaluate.train_file_partitioner import KFoldCrossPartitioner
# models_names = {"ag_news": "../FastTextModels/ag_news/ag_news.train",
#                 "dbpedia": "../FastTextModels/dbpedia/dbpedia.train",
#                 "yelp": "../FastTextModels/yelp_review_full/yelp_review_full.train"}

# ag_news = KFoldCrossPartitioner(5,
#                                 models_names["ag_news"],
#                                 "./folds_ag_news/",
#                                 "ag_news")
# ag_news.k_fold_cross_partitioner()
#
# dbpedia_folds = KFoldCrossPartitioner(5,
#                                       models_names["dbpedia"],
#                                       "./folds_dbpedia/",
#                                       "dbpedia")
# dbpedia_folds.k_fold_cross_partitioner()
#
# yelp_folds = KFoldCrossPartitioner(5,
#                                       models_names["yelp"],
#                                       "./folds_yelp/",
#                                       "yelp")
#
# yelp_folds.k_fold_cross_partitioner()

def fastText_evaluate(train_path, test_path, hyperparameters_set):
    start = time()
    to_valid_model = fstTxt.train_supervised(train_path, **hyperparameters_set)
    result = to_valid_model.test(test_path)
    end = time()
    return {"precision": result[1], "recall": result[2], "cpu_time": end - start}


if __name__ == '__main__':
    """experiments 1 default hyperparameters"""
    for model in ["ag_news", "dbpedia", "yelp"]:
        results = []
        for i in range(10):
            lkf = LocalKFoldCrossValidation(5, model, ("precision", "recall", "cpu_time"), fastText_evaluate)
            results.append(lkf.hyperparameters_kfc_parallel())

        parsed_results = []
        for iteration_result in results:
            parsed_results.append({**iteration_result["results"][0], 'total_completion_time': iteration_result['total_completion_time']})

        f = open("./marc_experiments/exp_1_" + model + ".csv", 'w')
        writer = csv.DictWriter(f, fieldnames=["precision", "recall", "cpu_time", "total_completion_time"])
        writer.writeheader()
        writer.writerows(parsed_results)
        f.close()

    """experiments 2 random search hyperparameters"""
    for number_of_sets in [5, 10, 20, 40, 80]:
        hyperparameters_sets = random_search(number_of_sets)
        for model in ["ag_news", "dbpedia", "yelp"]:
            results = []
            for i in range(2):
                lkf = LocalKFoldCrossValidation(5, model, ("precision", "recall", "cpu_time"), fastText_evaluate)
                results.append(lkf.hyperparameters_kfc_parallel(hyperparameters_sets))

            parsed_results = []
            for iteration_result in results:
                for set_index, set_res in enumerate(iteration_result["results"]):
                    parsed_results.append(
                        {"set_index": set_index, **set_res, 'total_completion_time': iteration_result['total_completion_time']})

            f = open("./marc_experiments/exp_2_hyperSets_" + str(number_of_sets) + "_" + model + ".csv", 'w')
            writer = csv.DictWriter(f, fieldnames=["set_index", "precision", "recall", "cpu_time", "total_completion_time"])
            writer.writeheader()
            writer.writerows(parsed_results)
            f.close()
