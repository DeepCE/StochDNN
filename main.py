import pandas as pd
import numpy as np
from ai import Neural
from lib import PatternReconigzer, Graph, differences, model_to_generate_trajectory, compute_errors, get_results, wasserstein
import random
import math


'''
###
IN
365

###
OUT
180
365

###
IN - OUT
1826
'''

MARKET = "gas"
METRICS = ["prices", "residual"]
DATASET_LEN = 15
SCALAR = 1000

END_DATE = "2023-06-30"

GEN_SERIES = 30
WINDOW = 1826

RE_GEN = True
PREFIX = "out"
STORAGE_PATH = 'data/%s_of_sample_%s_%s.csv' % (PREFIX, DATASET_LEN, WINDOW)

if __name__ == '__main__':
    # import tensorflow as tf
    # print(tf.test.is_built_with_cuda())
    # print(tf.config.list_physical_devices('GPU'))

    data_frame = pd.read_csv("data/detrend.csv")
    log_prices = data_frame["Price"].values
    log_ret = differences(log_prices)
    
    date_list = pd.date_range(end=END_DATE, periods=len(log_ret), freq='D')
    df = pd.DataFrame({"date": date_list, "prices": log_prices[1:], "residual":  log_ret})
    df = df[df["date"] >= "2017-07-01"]


    if PREFIX == "out":
        train_df = df[:len(df)-WINDOW]
        test_df = df[len(df)-WINDOW:]
        test_log_prices = test_df["prices"].values
    elif PREFIX == "in":
        train_df = df[:len(df)-WINDOW-365]
        test_df = df[len(df)-WINDOW-365:len(df)-365]
        test_log_prices = test_df["prices"].values
    else:
        print("Error")
        exit(0)

    if RE_GEN:
        neural = Neural("sim_%s" % DATASET_LEN, METRICS, DATASET_LEN)
        neural.load(schema=False)

        # use this call to analize generated series and check if where are repeated pattern
        pattern = PatternReconigzer(100)

        # load file with oredictions
        try: 
            gen_df = pd.read_csv(STORAGE_PATH)
            gen_df = gen_df.transpose()
            ai_matrix = gen_df.values.tolist()
        except Exception as e:
            print("File not found")
            ai_matrix = []

        # predict section
        while len(ai_matrix) < GEN_SERIES:
            print("*" * 60)
            print(len(ai_matrix))

            # create in_array
            in_array = []
            for p, r in zip(train_df["prices"].values[-DATASET_LEN+1:], train_df["residual"].values[-DATASET_LEN+1:]):
                in_array.append(p)
                in_array.append(r)
            in_array.append(test_df["prices"].values[0])
            in_array.append(test_df["residual"].values[0])

            # predictions
            MODEL = 3
            log_price_series = [test_log_prices[0]]
            probability_predictions = []

            for i in range(WINDOW-1):
                if MODEL == 1:
                    _TYPE = "log_ret"
                    index = 0 if _TYPE == "price" else 1
                    predicted_array = neural.predict(np.exp(in_array))[index][0]
                    array = []
                    for j, prob_class in enumerate(predicted_array):
                        if prob_class * 100 > 1:
                            for k in range(int(prob_class*100)):
                                array.append({"class": j, "probability": int(prob_class*100)})
                    predicted_class = random.choice(array)
                    probability_predictions.append(predicted_class["probability"])
                    last_price = in_array[-2]

                    if _TYPE == "price":
                        predicted_log_price = round(np.log(predicted_class["class"] / SCALAR), 3)
                        predicted_log_ret = predicted_log_price - last_price
                    else:
                        predicted_log_ret = round(np.log(predicted_class["class"] / SCALAR), 3)
                        predicted_log_price = last_price + predicted_log_ret
                else:
                    predicted_log_price = None
                    predicted_log_ret = None
                    for m, predicted_array in enumerate(neural.predict(np.exp(in_array))):
                        array = []
                        for j, prob_class in enumerate(predicted_array[0]):
                            if prob_class * 100 > 1:
                                for k in range(int(prob_class*100)):
                                    array.append({"class": j, "probability": int(prob_class*100)})
                        predicted_class = random.choice(array)
                        probability_predictions.append(predicted_class["probability"])
                        if m == 0:
                            predicted_log_price_1 = round(np.log(predicted_class["class"] / SCALAR), 3)
                        else:
                            predicted_log_ret_1 = round(np.log(predicted_class["class"] / SCALAR), 3)

                    last_price = in_array[-2]
                    predicted_log_price_2 = last_price + predicted_log_ret_1
                    predicted_log_price = (predicted_log_price_1 + predicted_log_price_2) / 2
                    predicted_log_ret = predicted_log_price - last_price

                in_array.remove(in_array[0])
                in_array.remove(in_array[0])                
                in_array.append(predicted_log_price)
                in_array.append(predicted_log_ret)

                # insert calculated price from predicted log_ret into series_prices
                log_price_series.append(predicted_log_price)

            if pattern.recognize(log_price_series) is False:
                print(np.mean(probability_predictions))
                if np.mean(probability_predictions) >= 50:
                    print("Stored serie")
                    ai_matrix.append(np.asarray(log_price_series))
                else:
                    print("Under treshold")
            else:
                print("Pattern detected")
                
        gen_df = pd.DataFrame(ai_matrix)
        gen_df = gen_df.transpose()
        gen_df.to_csv(STORAGE_PATH, index=False)
    else:
        gen_df = pd.read_csv(STORAGE_PATH)
        gen_df = gen_df.transpose()
        ai_matrix = gen_df.values.tolist()
    print("Founded %s series" % len(ai_matrix))

    model_ll_matrix = model_to_generate_trajectory("ll", n_traj=10000, n_obs=WINDOW, first_log_price=test_log_prices[0])
    model_mm_matrix = model_to_generate_trajectory("mm", n_traj=10000, n_obs=WINDOW, first_log_price=test_log_prices[0])

   
    '''
    ERRORS
    '''
    # errors for AI and models
    errors_values_dic = {"mae": {"ai": [], "ll": [], "mm": []}, "rmse": {"ai": [], "ll": [], "mm": []}, "rmse2": {"ai": [], "ll": [], "mm": []}}
    days_array = list(range(5, WINDOW, 5))
    if WINDOW not in days_array: days_array.append(WINDOW)

    for matrix, sign in zip([ai_matrix, model_ll_matrix, model_mm_matrix], ["ai", "ll", "mm"]):
        errors_mae_matrix = compute_errors(test_log_prices, matrix, "mae")
        errors_mae_matrix = np.asarray(errors_mae_matrix)
        errors_square_matrix = compute_errors(test_log_prices, matrix, "rmse")
        errors_square_matrix = np.asarray(errors_square_matrix)

        for i in days_array:
            #mae
            errors_values_dic["mae"][sign].append(np.mean(errors_mae_matrix[:, :i]))
            # square
            error = math.sqrt(np.mean(errors_square_matrix[:, :i]))
            errors_values_dic["rmse"][sign].append(error)
            # square 2
            series_error = []
            for array in errors_square_matrix:
                series_error.append(math.sqrt(np.mean(array[:i])))
            errors_values_dic["rmse2"][sign].append(np.mean(series_error))
        
    '''
    GRAPH
    '''
    graph1 = Graph("graph/%s", mode="png")
    graph1.generated([ai_matrix[0], model_ll_matrix[0], model_mm_matrix[0]], test_log_prices, "%s_%s_%s" % (PREFIX, DATASET_LEN, WINDOW))
    graph2 = Graph("errors_graph/%s", mode="all")
    graph2.errors(errors_values_dic, "%s_%s_%s" % (PREFIX, DATASET_LEN, WINDOW), days_array)
    graph3 = Graph("ai_graph/%s", mode="png")
    graph3.generated([ai_matrix[0], ai_matrix[1], ai_matrix[2]], test_log_prices, "%s_%s_%s" % (PREFIX, DATASET_LEN, WINDOW))
    
    # Stats properties computed on 1096 observations
    if WINDOW == 1826:
        for i, matrix in enumerate([ai_matrix, model_ll_matrix, model_mm_matrix]):
            print("-" * 60)

            if i == 0:
                print("AI")
            elif i == 1:
                print("LL")
            else:
                print("MM")
                
            stats_res = {"kt": [], "st": [], "sk": [], "mn": []}
            for series in matrix:
                kt, st, sk, mn = get_results(differences(series))
                stats_res["kt"].append(kt)
                stats_res["st"].append(st)
                stats_res["sk"].append(sk)
                stats_res["mn"].append(mn) 
            for prop in stats_res:
                print("%s - mean: %s | std: %s" % (prop.upper(), round(np.mean(stats_res[prop]), 4), round(np.std(stats_res[prop]), 4)))
            
            print("wasserstein: %s" % wasserstein(matrix, df["residual"].values))