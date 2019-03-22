import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate.localGenericHyperparameterEvaluate import *
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


"""experiments 1"""
if __name__ == '__main__':
    for model in ["ag_news", "dbpedia", "yelp"]:
        results = []
        for i in range(10):
            lkf = LocalKFoldCrossValidation(5, model, ("precision", "recall", "cpu_time"), fastText_evaluate)
            results.append(lkf.hyperparameters_kfc_parallel())

        parsed_results = []
        for inner_res in results:
            parsed_results.append({**inner_res["results"][0], **{'total_completion_time': inner_res['total_completion_time']}})

        f = open("./marc_experiments/exp_1_" + model + ".csv", 'w')
        writer = csv.DictWriter(f, fieldnames=["precision", "recall", "cpu_time", "total_completion_time"])
        writer.writeheader()
        writer.writerows(parsed_results)
        f.close()
