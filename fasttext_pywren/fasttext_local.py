import fastText as fstTxt
from time import time

ag_news_model = "./ag_news/ag_news.ftz"  # env arrange
amazon_review_model = "./amazon_review_polarity/amazon_review_polarity.ftz"  # env arrange
dbpedia_model = "./dbpedia/dbpedia.ftz"  # env arrange
sogou_news_model = "./sogou_news/sogou_news.ftz"  # env arrange
yelp_review_model = "./yelp_review_full/yelp_review_full.ftz"  # env arrange

bucketname = 'fasttext-predict-bucket'

files_names = {"ag_news": './ag_news/ag_news_predict.txt',
               "dbpedia": './dbpedia/dbpedia_predict.txt',
               "sogou_news": './sogou_news/sogou_predict.txt',
               "yelp_review_predict": './yelp_review_full/yelp_review_predict.txt'}


def ag_news_predictions():
    fasttext_model = fstTxt.load_model(ag_news_model)
    input_file = open(files_names['ag_news'], 'r').read()

    for line in input_file.splitlines():
        fasttext_model.predict(line)


def dbpedia_predictions():
    fasttext_model = fstTxt.load_model(dbpedia_model)
    input_file = open(files_names['dbpedia'], 'r').read()

    for line in input_file.splitlines():
        fasttext_model.predict(line)


def yelp_predictions():
    fasttext_model = fstTxt.load_model(yelp_review_model)
    input_file = open(files_names['yelp'], 'r').read()

    for line in input_file.splitlines():
        fasttext_model.predict(line)


def sogou_news_predictions():
    fasttext_model = fstTxt.load_model(sogou_news_model)
    input_file = open(files_names['sogou_news'], 'r').read()

    for line in input_file.splitlines():
        fasttext_model.predict(line)


start = time()

# ag_news_predictions()
# dbpedia_predictions()
# yelp_predictions()
sogou_news_predictions()

end = time()
duration = end - start
print("\nDuration: " + str(duration) + " Sec")
