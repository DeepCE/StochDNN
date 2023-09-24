import pandas as pd
import numpy as np
from ai import Neural
from lib import differences, get_results, PatternReconigzer, Graph
import tensorflow as tf
import random



MARKET = "gas"
METRICS = ["prices", "residual"]
DATASET_LEN = 10

SCALAR = 1000
GEN_SERIES = 1

END_DATE = "2023-06-30"

RE_GEN = True


if __name__ == '__main__':
    print(tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))

    data_frame = pd.read_csv("data/detrend.csv")
    log_prices = data_frame["Price"].values
    log_ret = differences(log_prices)
    # WINDOW = len(log_prices)
    WINDOW = 5000

    if RE_GEN:
        date_list = pd.date_range(end=END_DATE, periods=len(log_ret), freq='D')
        df = pd.DataFrame({"date": date_list, "prices": log_prices[1:], "residual":  log_ret})
        # df = df[df["date"] >= "2017-07-01"]
        df[df["date"] >= "2023-06-01"]

        neural = Neural(MARKET, METRICS, DATASET_LEN)
        neural.load()

        # use this call to analize generated series and check if where are repeated pattern
        pattern = PatternReconigzer()

        # predict section
        seires_matrix = []
        stats_res = {"kt": [], "st": [], "sk": [], "mn": []}
        while len(seires_matrix) != GEN_SERIES:
            print(len(seires_matrix))

            # create in_array
            in_array = []
            for p, r in zip(df["prices"].values[:DATASET_LEN], df["residual"].values[:DATASET_LEN]):
                in_array.append(p)
                in_array.append(r)

            # predictions
            log_price_series = [df["prices"].values[0]]
            for i in range(WINDOW-1):
                predicted_array = neural.predict(np.exp(in_array))[0]

                # randomize selected class
                array = []
                for j, prob_class in enumerate(predicted_array):
                    if prob_class * 100 > 1:
                        for k in range(int(prob_class*100)):
                            array.append(j)
                predicted_class = random.choice(array)
                predicted_log_price = round(np.log(predicted_class / SCALAR), 3)

                # remove first 2 elements
                in_array.remove(in_array[0])
                in_array.remove(in_array[0])

                last_price = in_array[-2]
                predicted_log_ret = predicted_log_price - last_price
                in_array.append(predicted_log_price)
                in_array.append(predicted_log_ret)

                # insert calculated price from predicted log_ret into series_prices
                log_price_series.append(predicted_log_price)

            if pattern.recognize(log_price_series) is False:
                seires_matrix.append(np.asarray(log_price_series))
                kt, st, sk, mn = get_results(differences(log_price_series))
                stats_res["kt"].append(kt)
                stats_res["st"].append(st)
                stats_res["sk"].append(sk)
                stats_res["mn"].append(mn)

        
        gen_df = pd.DataFrame(seires_matrix)
        gen_df = gen_df.transpose()
        gen_df.to_csv('data/6_years.csv', index=False)
    else:
        gen_df = pd.read_csv('data/6_years.csv')
        gen_df = gen_df.transpose()
        seires_matrix = gen_df.values.tolist()
    
    print(get_results(log_ret))
    for prop in stats_res:
        print("%s: %s" % (prop, np.mean(stats_res[prop])))

    graph = Graph(mode="png")
    if len(seires_matrix) == 1:
        seires_matrix.append([])
    graph.rebuild(seires_matrix, log_prices, "6_years")