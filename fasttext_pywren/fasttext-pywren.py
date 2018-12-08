from fastText import *
import fasttext_pywren.downloaded_models.amazon_review_full.label_map as label_mapping



# file_2_predict = open("./amazon_review_full/amazon_review_full.predict", 'r')
# fasttext_model = load_model("./amazon_review_full/amazon_review_full.bin")
#
# for to_predict in file_2_predict.read().splitlines():
#     labels, prob_arr = fasttext_model.predict(to_predict, 1)
#     label_str = ""
#     for label in labels:
#         label_str += label_mapping.label_2_text.get(label) + " "
#     print(label_str)

fasttext_model = train_supervised("./amazon_real_reviews_model/database/Movies_and_TV.train")

print(fasttext_model.test("./amazon_real_reviews_model/database/Movies_and_TV.test"))