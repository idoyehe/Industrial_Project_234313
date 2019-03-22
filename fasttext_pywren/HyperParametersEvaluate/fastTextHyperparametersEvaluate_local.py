import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate.localGenericHyperparameterEvaluate import *
import csv

models_names = {"dbpedia": "../FastTextModels/dbpedia/dbpedia.train",
                "yelp": "../FastTextModels/yelp_review_full/yelp_review_full.train"}


def fastText_evaluate(train_path, test_path, hyperparameters_set):
    to_valid_model = fstTxt.train_supervised(train_path, **hyperparameters_set)
    result = to_valid_model.test(test_path)
    return {"precision": result[1], "recall": result[2]}




"""experiments 1"""
if __name__ == '__main__':
    results = []
    for i in range(5):
        lkf = LocalKFoldCrossValidation(5, models_names['dbpedia'], ("precision", "recall"), fastText_evaluate)
        results.append(lkf.hyperparameters_kfc())

    parsed_results = []
    for inner_res in results:
        completion_time = {"completion_time": inner_res["completion_time"]}
        parsed_results.append({**inner_res["results"][0], **completion_time})

    f = open("./marc_experiments/exp_1_dbpedia.csv", 'w')
    writer = csv.DictWriter(f, fieldnames=["precision", "recall", "validation_completion_time","completion_time"])
    writer.writeheader()
    writer.writerows(parsed_results)
    f.close()

    results = []
    for i in range(5):
        lkf = LocalKFoldCrossValidation(5, models_names['yelp'], ("precision", "recall"), fastText_evaluate)
        results.append(lkf.hyperparameters_kfc())

    parsed_results = []
    for inner_res in results:
        completion_time = {"completion_time": inner_res["completion_time"]}
        parsed_results.append({**inner_res["results"][0], **completion_time})

    f = open("./marc_experiments/exp_1_yelp.csv", 'w')
    writer = csv.DictWriter(f, fieldnames=["precision", "recall", "validation_completion_time","completion_time"])
    writer.writeheader()
    writer.writerows(parsed_results)
    f.close()
