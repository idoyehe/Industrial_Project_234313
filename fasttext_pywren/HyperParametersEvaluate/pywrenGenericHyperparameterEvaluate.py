import pywren_ibm_cloud as pywren
import types
import os
from time import time


class PywrenHyperParameterUtil(object):
    def __init__(self, evaluate_learning_algo_function: types.FunctionType, bucket_name: str, file_key: str,
                 job_name: str = None, graphs_path: str = None, path_docker: str = None):
        """
        :param job_name: job name for invocation graphs
        :param evaluate_learning_algo_function: function to evaluate model with given parameters
            it signature should be  (parameters_dict, train_path, test_path) and return {"precision": precision, "recall": recall}
        :param bucket_name: the bucket name stores train file
        :param file_key: train file name as stores in bucket given bucket
        :param path_docker: specific folder to save temporary files, can be none and will create while run time
        """
        self.file_key = file_key
        self.bucket_name = bucket_name
        self.evalute_learning_algo_function = evaluate_learning_algo_function
        self.path_docker = path_docker
        self.k_value = None
        self.path_docker_default = "/hyperParametersTuning/"
        self.job_name = job_name
        self.graphs_path = graphs_path
        self.parameters_list = None
        self.eval_keys = None
        self.train_duration_label = "trainDuration"
        self.runtime = None

    def set_kvalue(self, mew_k_value: int):
        """
        :param mew_k_value: new k value to preform k fold cross validation
        :return: nothing
        """
        if mew_k_value <= 1:
            raise Exception("K fold cross validation value should be 2 minimum")
        self.k_value = mew_k_value

    def set_evaluation_keys(self, eval_keys: list):
        """
        :param eval_keys: a list of keys (strings) of the dictionary that return from the learning algorithm function
        :return: nothing
        """
        if eval_keys is None or len(eval_keys) < 1:
            raise Exception("Please provide evaluation keys list with length minimum 1")
        self.eval_keys = eval_keys + [self.train_duration_label]

    def set_parameters(self, parameters_list: list):
        """
        :param parameters_list: list of parameters to evaluate
        :return:
        """
        if type(parameters_list) is not list:
            raise TypeError("parameters_list should be list object")

        self.parameters_list = parameters_list

    def evaluate_params(self, runtime="pywren_3.6"):
        """
        :param runtime: thr action in cloud to invoke
        :return: a dict that has 2 keys:
            Results list of evaluation of all set of parameters
            Duration the time takes to PyWren for execution
        """
        if self.k_value is None:
            raise Exception("Please call to set k value method and try again")

        if self.eval_keys is None:
            raise Exception("Please call to set evaluation keys method and try again")

        self.runtime = runtime
        start = time()
        pw = pywren.ibm_cf_executor(runtime=self.runtime)
        pw.map(self.__map_evaluate_parameters, self.parameters_list)
        pw_res = pw.get_result()
        duration = time() - start
        self.runtime = None
        if self.job_name and self.graphs_path:
            pw.create_timeline_plots(dst=self.graphs_path, name=self.job_name)
        return {"Results": pw_res, "TotalDuration": duration}

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
        for line in data_stream.read().splitlines():
            str_line = line.decode('utf-8')  # data comes from COS
            str_line += "\n"
            if current_index % self.k_value == valid_index:
                valid_file.write(str_line)
            else:
                train_file.write(str_line)

            current_index += 1

        valid_file.close()
        train_file.close()

        start = time()
        evaluation_res = self.evalute_learning_algo_function(params_dict, train_path, valid_path)
        end = time()
        evaluation_res[self.train_duration_label] = end - start

        return evaluation_res

    def __reducer_average_validator(self, results):
        k_cross_valid_dict = {}
        for key in self.eval_keys:  # initialize all sums to 0
            k_cross_valid_dict[key] = 0

        for res in results:
            for key in self.eval_keys:  # adding to all sums
                k_cross_valid_dict[key] += res[key]

        for key in self.eval_keys:  # average each sum
            k_cross_valid_dict[key] /= len(results)
        return k_cross_valid_dict

    def __map_evaluate_parameters(self, parameters_dict, ibm_cos):
        k_list = [i for i in range(self.k_value)]

        def __k_fold_cross_validation_wrap(valid_index, ibm_cos):
            file = ibm_cos.get_object(Bucket=self.bucket_name, Key=self.file_key)
            data_stream = file['Body']
            return self.__map_k_fold_cross_validation(data_stream, parameters_dict, valid_index)

        def __reducer_average_validator_wrap(results):
            return self.__reducer_average_validator(results)

        sub_pw = pywren.ibm_cf_executor(runtime=self.runtime)
        sub_pw.map_reduce(__k_fold_cross_validation_wrap, k_list, __reducer_average_validator_wrap, reducer_wait_local=False)
        return sub_pw.get_result()
