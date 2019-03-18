import pywren_ibm_cloud as pywren
import types
from os import mkdir, path
from shutil import rmtree
from sklearn.model_selection import KFold
from time import time


class PywrenHyperParameterUtil(object):
    def __init__(self, evaluate_learning_algo_function: types.FunctionType, bucket_name: str, file_key: str,
                 job_name: str = None, graphs_path: str = None):
        """
        :param job_name: job name for invocation graphs
        :param evaluate_learning_algo_function: function to evaluate model with given parameters
            it signature should be  (parameters_dict, train_path, test_path) and return {"precision": precision, "recall": recall}
        :param bucket_name: the bucket name stores train file
        :param file_key: train file name as stores in bucket given bucket
        """
        self.file_key = file_key
        self.bucket_name = bucket_name
        self.evalute_learning_algo_function = evaluate_learning_algo_function
        self.k_value = None
        self.path_docker_default = "/hyperParametersKFC/"
        self.folders_prefix = "fold_"
        self.train_file_name = "database.train"
        self.test_file_name = "database.test"
        self.job_name = job_name
        self.graphs_path = graphs_path
        self.parameters_list = None
        self.eval_keys = None
        self.validation_time_label = "validation_completion_time"
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
        self.eval_keys = eval_keys + [self.validation_time_label]

    def set_parameters(self, parameters_list: list):
        """
        :param parameters_list: list of parameters to evaluate
        :return:
        """
        if type(parameters_list) is not list:
            raise TypeError("parameters_list should be list object")

        self.parameters_list = parameters_list

    def kfcv_partitioner(self, ibm_cos):
        file = ibm_cos.get_object(Bucket=self.bucket_name, Key=self.file_key)
        labeled_data = file['Body'].read().splitlines()
        kf = KFold(n_splits=self.k_value, shuffle=True)
        splits = kf.split(labeled_data)

        i: int = 0
        for train_index, test_index in splits:
            train_name = "fold_" + str(i) + ".train"
            test_name = "fold_" + str(i) + ".test"
            i += 1
            train_data = '\n'.join([labeled_data[index].decode('utf-8') for index in train_index])
            test_data = '\n'.join([labeled_data[index].decode('utf-8') for index in test_index])
            ibm_cos.put_object(Bucket=self.bucket_name, Key=train_name, Body=train_data)
            ibm_cos.put_object(Bucket=self.bucket_name, Key=test_name, Body=test_data)
        return

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
        pw.call_async(self.kfcv_partitioner, [])
        results = pw.get_result()

        # pw = pywren.ibm_cf_executor(runtime=self.runtime)
        # pw.map(self.__map_evaluate_parameters, self.parameters_list)
        # pw_res = pw.get_result()
        completion_time = time() - start
        return {"Results": results, "total_completion_time": completion_time}

    def __map_k_fold_cross_validation(self, params_dict, index):

        train_file: str = self.path_docker_default + self.folders_prefix + str(index) + "/" + self.train_file_name
        test_file: str = self.path_docker_default + self.folders_prefix + str(index) + "/" + self.test_file_name

        start = time()
        evaluation_res = self.evalute_learning_algo_function(train_file, test_file, params_dict)
        evaluation_res[self.validation_time_label] = time() - start

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

    def __map_evaluate_parameters(self, parameters_dict):
        k_list = [i for i in range(self.k_value)]

        def __k_fold_cross_validation_wrap(valid_index):
            return self.__map_k_fold_cross_validation(parameters_dict, valid_index)

        def __reducer_average_validator_wrap(results):
            return self.__reducer_average_validator(results)

        # sub_pw = pywren.ibm_cf_executor(runtime=self.runtime)
        # sub_pw.map_reduce(__k_fold_cross_validation_wrap, k_list, __reducer_average_validator_wrap, reducer_wait_local=False)
        # sub_pw.call_async(files_checker, [])
        return files_checker()
