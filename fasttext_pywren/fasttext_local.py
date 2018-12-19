import fastText as fstTxt
from time import time
from sys import getsizeof

ag_news_model = "./ag_news/ag_news.ftz"  # env arrange
amazon_review_model = "./amazon_review_polarity/amazon_review_polarity.ftz"  # env arrange
dbpedia_model = "./dbpedia/dbpedia.ftz"  # env arrange
sogou_news_model = "./sogou_news/sogou_news.ftz"  # env arrange
yelp_review_model = "./yelp_review_full/yelp_review_full.ftz"  # env arrange

bucketname = 'fasttext-predict-bucket'

files_names = {"ag_news": './ag_news/ag_news_predict.txt',  # ~ 4 sec
               "dbpedia": './dbpedia/dbpedia_predict.txt',  # ~ 23 sec
               "sogou_news": './sogou_news/sogou_predict.txt',  # ~ 140 sec
               "yelp": './yelp_review_full/yelp_review_predict.txt'}  # ~  50 sec


def ag_news_predictions():
    fasttext_model = fstTxt.load_model(ag_news_model)
    input_file = open(files_names['ag_news'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        label, prob = fasttext_model.predict(line)
        result.append((label[0], prob[0]))
    return result


def dbpedia_predictions():
    fasttext_model = fstTxt.load_model(dbpedia_model)
    input_file = open(files_names['dbpedia'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        label, prob = fasttext_model.predict(line)
        result.append((label[0], prob[0]))
    return result


def yelp_predictions():
    fasttext_model = fstTxt.load_model(yelp_review_model)
    input_file = open(files_names['yelp'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        label, prob = fasttext_model.predict(line)
        result.append((label[0], prob[0]))
    return result


def sogou_news_predictions():
    fasttext_model = fstTxt.load_model(sogou_news_model)
    input_file = open(files_names['sogou_news'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        label, prob = fasttext_model.predict(line)
        result.append((label[0], prob[0]))
    return result


start = time()

results = ag_news_predictions()
# results = dbpedia_predictions()
# results = yelp_predictions()
# results = sogou_news_predictions()

end = time()
duration = end - start
print("\nDuration: " + str(duration) + " Sec")
print("\nResult Size: " + str(getsizeof(results)) + " Bytes")