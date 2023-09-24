import pandas as pd
import numpy as np
from ai import Neural
from lib import differences, get_results, create_series_dataset, create_single_series_dataset
import tensorflow as tf


if __name__ == '__main__':
    print(tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))

    MARKET = "gas"

    METRICS = ["prices", "residual"]
    DATASET_LEN = 15
    EPOCHS = 900
    BATCH = 64

    WINDOW = 365
    SCALAR = 1000
    TRAIN = 1
    N_OUTPUT = 2

    data_frame = pd.read_csv("data/detrend.csv")
    log_prices = data_frame["Price"].values
    log_ret = differences(log_prices)
    print(get_results(log_ret))

    df = pd.DataFrame({"prices": log_prices[1:], "residual":  log_ret})
    train_df = df[:len(df)-WINDOW]
    test_df = df[len(df)-WINDOW:]
    test_log_prices = test_df["prices"].values

    if N_OUTPUT == 1:
        x_train, y_train = create_single_series_dataset(np.exp(train_df), METRICS, DATASET_LEN, SCALAR, "prices")
    else:
        x_train, y_train = create_series_dataset(np.exp(train_df), METRICS, DATASET_LEN, SCALAR)

    # Configura le opzioni di TensorFlow per utilizzare la GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as session:
        # create neural object
        neural = Neural("sim_%s" % DATASET_LEN, METRICS, DATASET_LEN, n_output=N_OUTPUT)
        if TRAIN == 1:
            neural.train(x_train, y_train, epochs=EPOCHS, batch=BATCH)
        elif TRAIN == 2:
            neural.resume_train(x_train, y_train, epochs=EPOCHS, batch=BATCH)