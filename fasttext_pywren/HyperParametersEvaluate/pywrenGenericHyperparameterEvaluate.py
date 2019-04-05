import pywren_ibm_cloud as pywren
import types
from time import time


class PywrenHyperParameterUtil(object):
    def __init__(self, evaluate_learning_algo_function: types.FunctionType, bucket_name: str, model_name: str,
                 job_name: str = None, local_graphs_path: str = None):

        self.model_name = model_name
        self.bucket_name = bucket_name
        self.evalute_learning_algo_function = evaluate_learning_algo_function
        self.k_value = None
        self.path_docker_default = "/hyperParametersKCFV/"
        self.job_name = job_name
        self.local_graphs_path = local_graphs_path
        self.hyperparameters_list = None
        self.evaluate_keys = None
        self.total_completion_time = "total_completion_time"
        self.runtime = None
        self.runtime_memory = None

    def set_kvalue(self, mew_k_value: int):
        """
        :param mew_k_value: new k value to preform k fold cross validation
        :return: nothing
        """
        if mew_k_value < 2:
            raise Exception("K fold cross validation value should be minimum 2")
        self.k_value = mew_k_value

    def set_evaluation_keys(self, evaluate_keys: tuple):
        """
        :param evaluate_keys: a list of keys (strings) of the dictionary that return from the learning algorithm function
        :return: nothing
        """
        if evaluate_keys is None or len(evaluate_keys) < 1:
            raise Exception("Please provide evaluation keys tuple with length minimum 1")
        self.evaluate_keys = evaluate_keys

    def set_parameters(self, hyperparameters_list: list):
        """
        :param hyperparameters_list: list of parameters to evaluate
        :return:
        """
        if type(hyperparameters_list) is not list:
            raise TypeError("parameters_list should be list object")

        self.hyperparameters_list = hyperparameters_list

    def evaluate_params(self, runtime: str = "ibmcloudfunction/action-pywren-v3.6", runtime_memory: int = 2048):
        """
        :param runtime: the action in cloud to invoke
        :param runtime_memory: memory to use in the runtime
        :return: a dict that has 2 keys:
            Results list of evaluation of all set of parameters
            Duration the time takes to PyWren for execution
        """
        if self.k_value is None:
            raise Exception("Please call to set k value method and try again")

        if self.evaluate_keys is None:
            raise Exception("Please call to set evaluation keys method and try again")

        self.runtime = runtime
        self.runtime_memory = runtime_memory
        start = time()
        pywren_executor = pywren.ibm_cf_executor(runtime=self.runtime, runtime_memory=self.runtime_memory)
        pywren_executor.map(self.__map_evaluate_hyperparameters, self.hyperparameters_list)
        pywren_results = pywren_executor.get_result()
        total_completion_time = time() - start
        if self.job_name and self.local_graphs_path:
            pywren_executor.create_timeline_plots(dst=self.local_graphs_path, name=self.job_name)
        if not isinstance(pywren_results, list):  # wrapped in list for single evaluation
            pywren_results = [pywren_results]
        return {"results": pywren_results, "total_completion_time": total_completion_time}

    def __map_evaluate_hyperparameters(self, parameters_dict, ibm_cos):
        folds_indexes = list(range(self.k_value))

        def __k_fold_cross_validation_wrap(fold_index, ibm_cos):
            cos_train_key: str = self.model_name + "_fold_" + str(fold_index) + ".train"
            cos_test_key: str = self.model_name + "_fold_" + str(fold_index) + ".test"
            local_train_key = self.path_docker_default + cos_train_key
            local_test_key = self.path_docker_default + cos_test_key

            cos_train_file = ibm_cos.get_object(Bucket=self.bucket_name, Key=cos_train_key)
            cos_test_file = ibm_cos.get_object(Bucket=self.bucket_name, Key=cos_test_key)

            dec = lambda l: l.decode('utf-8') + '\n'

            local_train_file = cos_train_file['Body'].read().splitlines()
            local_train_file = list(map(dec, local_train_file))
            local_test_file = cos_test_file['Body'].read().splitlines()
            local_test_file = list(map(dec, local_test_file))

            train_file = open(local_train_key, 'w')
            test_file = open(local_test_key, 'w')
            train_file.writelines(local_train_file)
            test_file.writelines(local_test_file)
            test_file.close()
            train_file.close()
            return self.evalute_learning_algo_function(local_train_key, local_test_key, parameters_dict)

        def __reducer_average_validator_wrap(results):
            k_cross_valid_dict = {}
            for key in self.evaluate_keys:  # initialize all sums to 0
                k_cross_valid_dict[key] = 0.0

            for res in results:
                for key in self.evaluate_keys:  # adding to all sums
                    k_cross_valid_dict[key] += res[key]

            for key in self.evaluate_keys:  # average each sum
                k_cross_valid_dict[key] /= len(results)
            return k_cross_valid_dict

        nested_pywren_executor = pywren.ibm_cf_executor(runtime=self.runtime, runtime_memory=self.runtime_memory)
        nested_pywren_executor.map(map_function=__k_fold_cross_validation_wrap,
                                          map_iterdata=folds_indexes)
        nested_pywren_results = nested_pywren_executor.get_result()

        return __reducer_average_validator_wrap(nested_pywren_results)
