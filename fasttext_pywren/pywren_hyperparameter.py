import pywren_ibm_cloud as pywren
import types
import os
from time import time


class PywrenHyperParameterUtil(object):
    def __init__(self, evalute_learning_algo_function: types.FunctionType, bucket_name: str, file_key: str, k_value: int = 5,
                 path_docker: str = None):
        self.file_key = file_key
        self.bucket_name = bucket_name
        self.evalute_learning_algo_function = evalute_learning_algo_function
        self.path_docker = path_docker
        self.k_value = k_value
        self.path_docker_default = "/hyperParametersTuning/"
        self.parameters_list = None

    def set_kvalue(self, mew_k_value: int):
        self.k_value = mew_k_value
        # if self.parameters_list is list:
        #     for index, params in enumerate(self.parameters_list):
        #         self.parameters_list[index] = (self.k_value, params)

    def set_parameters(self, parameters_list: list):
        self.parameters_list = parameters_list
        # self.parameters_list = list()
        # for params in parameters_list:
        #     self.parameters_list.append((self.k_value, params))

    def evalute_params(self, runtime="pywren_3.6"):
        start = time()
        pw = pywren.ibm_cf_executor(runtime=runtime)
        pw.map(self.__map_evaluate_parameters, self.parameters_list)
        pw_res = pw.get_result()
        duration = time() - start
        return {"Results: ": pw_res, "Duration": duration}

    def __map_k_fold_cross_validation(self, data_stream, params_dict, valid_index):
        if self.path_docker is None:
            if os.path.exists(self.path_docker_default) is False:
                os.mkdir(self.path_docker_default)
            self.path_docker = self.path_docker_default

        valid_path = self.path_docker + "source.valid"
        train_path = self.path_docker + "source.train"

        train_file = open(train_path, 'w')
        valid_file = open(valid_path, 'w')

        current_index = 0
        for line in data_stream.splitlines():
            str_line = line  # line.decode('utf-8')  # in case data comes from COS
            str_line += "\n"
            if current_index % self.k_value == valid_index:
                valid_file.write(str_line)
            else:
                train_file.write(str_line)

            current_index += 1

        valid_file.close()
        train_file.close()

        return self.evalute_learning_algo_function(params_dict, train_path, valid_path)

    def __map_evaluate_parameters(self, lr, dim, ws, epoch, minCount):

        parameters_dict = {"lr": lr, "dim": dim, "ws": ws, "epoch": epoch, "minCount": minCount}

        k_list = [i for i in range(self.k_value)]

        def k_cross_validtion_wrap(valid_index):
            data_stream = open("/fasttext/validation/dbpedia.train", "r").read()  # should be with COS
            return self.__map_k_fold_cross_validation(data_stream, parameters_dict, valid_index)

        def reducer_average_validator(results):
            avg_precision = 0
            avg_recall = 0
            for res in results:
                avg_precision += res["precision"]
                avg_recall += res["recall"]

            avg_precision /= len(results)
            avg_recall /= len(results)
            return {"precision": avg_precision, "recall": avg_recall}

        sub_pw = pywren.ibm_cf_executor(runtime="fasttext-hyperparameter")
        sub_pw.map_reduce(k_cross_validtion_wrap, k_list, reducer_average_validator, reducer_wait_local=False)
        return sub_pw.get_result()
