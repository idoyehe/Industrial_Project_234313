import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate import pywrenGenericHyperparameterEvaluate
from fasttext_pywren.HyperParametersEvaluate.randomHyperparameters import random_search
import csv

bucket_name = 'fasttext-train-datasets'

"""evaluation model function instance"""


def fastText_evaluate(train_path, test_path, hyperparameters_set):
    from time import time
    start = time()
    to_valid_model = fstTxt.train_supervised(train_path, **hyperparameters_set)
    result = to_valid_model.test(test_path)
    end = time()
    return {"precision": result[1], "recall": result[2], "cpu_time": end - start}


if __name__ == '__main__':
    """experiments 1 default hyperparameters"""
    for model in ["ag_news", "dbpedia"]:  # , "yelp"]: causing out of memory
        results = []
        for i in range(10):
            """call for PyWren exaction with list of hyperparameters"""
            pywren_kcfv = pywrenGenericHyperparameterEvaluate.PywrenHyperParameterUtil(
                fastText_evaluate,
                bucket_name,
                model,
                job_name=model + "_number_of_sets_1_iter_" + str(i),
                local_graphs_path="./InvocationsGraphsFiles/")

            pywren_kcfv.set_kvalue(5)
            pywren_kcfv.set_evaluation_keys(("precision", "recall", "cpu_time"))
            pywren_kcfv.set_parameters([[{}]])
            print("model is: " + model + ", iteration #: " + str(i))
            results.append(pywren_kcfv.evaluate_params(runtime="idoye/fasttext-hyperparameters_3.7"))
        parsed_results = []
        for iteration_result in results:
            parsed_results.append({**iteration_result["results"][0], 'total_completion_time': iteration_result['total_completion_time']})

        f = open("./marc_experiments/pywren_exp_1_" + model + ".csv", 'w')
        writer = csv.DictWriter(f, fieldnames=["precision", "recall", "cpu_time", "total_completion_time"])
        writer.writeheader()
        writer.writerows(parsed_results)
        f.close()

    """experiments 2 random search hyperparameters pywren"""
    for number_of_sets in [5, 25, 50, 75]:
        hyperparameters_sets = random_search(number_of_sets)
        for model in ["ag_news", "dbpedia"]:  # , "yelp"]: causing out of memory
            results = []
            for i in range(5):
                """call for PyWren exaction with list of hyperparameters"""
                pywren_kcfv = pywrenGenericHyperparameterEvaluate.PywrenHyperParameterUtil(
                    fastText_evaluate,
                    bucket_name,
                    model,
                    job_name=model + "_number_of_sets_" + str(number_of_sets) + "_iter_" + str(i),
                    local_graphs_path="./InvocationsGraphsFiles/")

                pywren_kcfv.set_kvalue(5)
                pywren_kcfv.set_evaluation_keys(("precision", "recall", "cpu_time"))
                pywren_kcfv.set_parameters(hyperparameters_sets)
                print("model is: " + model + " # sets is: " + str(number_of_sets) + ", iteration #: " + str(i))
                results.append(pywren_kcfv.evaluate_params(runtime="idoye/fasttext-hyperparameters"))

            parsed_results = []
            for iteration_result in results:
                for set_index, set_res in enumerate(iteration_result["results"]):
                    parsed_results.append(
                        {"set_index": set_index, **set_res, 'total_completion_time': iteration_result['total_completion_time']})

            f = open("./marc_experiments/pywren_exp_2_hyperSets_" + str(number_of_sets) + "_" + model + ".csv", 'w')
            writer = csv.DictWriter(f, fieldnames=["set_index", "precision", "recall", "cpu_time", "total_completion_time"])
            writer.writeheader()
            writer.writerows(parsed_results)
            f.close()
