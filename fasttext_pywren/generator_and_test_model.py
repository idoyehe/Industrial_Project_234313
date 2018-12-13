from fastText import *
from time import time

model_label = "all_by_label"
train_file_path = "./database/" + model_label + ".train"
bin_file_path = "./database/" + model_label + ".bin"
test_file_path = "./database/" + model_label + ".test"
quantize_file_path = "./database/" + model_label + ".ftz"

generate_model = True
test_model = True
quantize_and_test_model = True

if generate_model:
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

if test_model:
    start_time = time()
    fasttext_model = load_model(bin_file_path)
    elapsed = time()
    duration = elapsed - start_time
    print("\nDuration to LOAD the model: " + str(duration) + " Sec")

    start_time = time()
    print(fasttext_model.test(test_file_path))
    elapsed = time()
    duration = elapsed - start_time
    print("\nDuration to TEST the model NOT Quantize: " + str(duration) + " Sec")

if quantize_and_test_model:
    start_time = time()
    fasttext_model.quantize()
    elapsed = time()
    duration = elapsed - start_time
    print("\nDuration to SAVE the model: " + str(duration) + " Sec")

    start_time = time()
    fasttext_model.save_model(quantize_file_path)
    elapsed = time()
    duration = elapsed - start_time
    print("\nDuration to SAVE the model: " + str(duration) + " Sec")

    start_time = time()
    print(fasttext_model.test(test_file_path))
    elapsed = time()
    duration = elapsed - start_time
    print("\nDuration to TEST the model Quantize: " + str(duration) + " Sec")
