from fastText import *
from time import time
import fasttext_pywren.amazon_real_reviews_model.label_map as review_labels

train_file_path = "./amazon_real_reviews_model/database/all_reviews.train"
bin_file_path = "./amazon_real_reviews_model/database/all_reviews.bin"
test_file_path = "./amazon_real_reviews_model/database/all_reviews.test"

start_time = time()
fasttext_model = train_supervised(train_file_path, lr=1.0, wordNgrams=5, loss="hs")
elapsed = time()
duration = elapsed - start_time
print("\nDuration to BUILD the model: " + str(duration) + " Sec")

start_time = time()
fasttext_model.save_model(bin_file_path)
elapsed = time()
duration = elapsed - start_time
print("\nDuration to SAVE the model: " + str(duration) + " Sec")

start_time = time()
fasttext_model = load_model(bin_file_path)
elapsed = time()
duration = elapsed - start_time
print("\nDuration to LOAD the model: " + str(duration) + " Sec")


start_time = time()
print(fasttext_model.test(test_file_path))
elapsed = time()
duration = elapsed - start_time
print("\nDuration to TEST the model: " + str(duration) + " Sec")