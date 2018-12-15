import fastText as fstTxt
import pywren_ibm_cloud as pywren

ag_news_model = "/fasttext/models/ag_news.ftz"  # from docker
amazon_review_model = "/fasttext/models/amazon_review_polarity.ftz"  # from docker
dbpedia_model = "/fasttext/models/dbpedia.ftz"  # from docker
sogou_news_model = "/fasttext/models/sogou_news.ftz"  # from docker
yelp_review_model = "/fasttext/models/yelp_review_full.ftz"  # from docker

bucketname = 'fasttext-predict-bucket'

files_names = ['ag_news_predict.txt',
               'dbpedia_predict.txt',
               'sogou_predict.txt',
               'yelp_review_predict.txt']

files_to_predict = list(map(lambda s: bucketname + '/' + s, files_names))


def map_fasttext_function(key, data_stream):
    print('I am processing the object {}'.format(key))
    fasttext_model = fstTxt.load_model(ag_news_model)

    data = data_stream.read()
    for line in data.splitlines():
        fasttext_model.predict(str(line))

    return True


def my_reduce_function(results):
    return True


chunk_size = 4 * 1024 ** 2  # 4MB

pw = pywren.ibm_cf_executor(runtime="fasttext-exists-models")
pw.map_reduce(map_fasttext_function, files_to_predict[0], my_reduce_function, chunk_size)
print(pw.get_result())
