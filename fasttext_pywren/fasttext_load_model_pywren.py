from time import time
import fastText as fstTxt
import pywren_ibm_cloud as pywren
from pickle import dump
from sys import getsizeof

ag_news_model = "/fasttext/models/ag_news.ftz"  # from docker
amazon_review_model = "/fasttext/models/amazon_review_polarity.ftz"  # from docker
dbpedia_model = "/fasttext/models/dbpedia.ftz"  # from docker
sogou_news_model = "/fasttext/models/sogou_news.ftz"  # from docker
yelp_review_model = "/fasttext/models/yelp_review_full.ftz"  # from docker

bucketname = 'fasttext-predict-bucket'

files_names = ['ag_news_predict.txt',  # 27.6 MB ~ 40 sec
               'dbpedia_predict.txt',  # 164.8 MB ~ 158.247 sec when chunk size is 4MB
               'sogou_predict.txt',  # 1.2 GB ~ 90 sec when chunk size is 16MB ~ 144 sec
               'yelp_review_predict.txt']  # 456.2 MB ~ 247 MB

files_to_predict = list(map(lambda s: bucketname + '/' + s, files_names))


def map_fasttext_function(key, data_stream):
    print('I am processing the object {}'.format(key))
    fasttext_model = fstTxt.load_model(dbpedia_model)

    data = data_stream.read()
    my_list = []
    for line in data.splitlines():
        my_list.append(fasttext_model.predict(str(line)))

    return my_list


def reduce_function(results):
    my_result = list()

    for my_list in results:
        my_result.extend(my_list)
    return {"futures": None, "results": getsizeof(my_result)}


chunk_size = 8 * 1024 ** 2  # 4MB

start = time()

pw = pywren.ibm_cf_executor(runtime="fasttext-exists-models")
pw.map_reduce(map_fasttext_function, files_to_predict[1], reduce_function, chunk_size=chunk_size)
result_object = pw.get_result()
futures = result_object['futures']
if futures is not None:
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]
    res = {'run_statuses': run_statuses, 'invoke_statuses': invoke_statuses}
    dump(res, open('../InvocationsGraphsFiles/statuses.pickle', 'wb'), -1)

end = time()
duration = end - start
print("\nDuration: " + str(duration) + " Sec")
print("\nResult Size: " + str(result_object['results']) + " Bytes")