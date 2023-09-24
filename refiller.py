import pandas as pd
import numpy as np
from ai import Neural
from lib import Graph, differences, get_results, create_single_series_dataset 


def reformat_date(value):
    a = value.split(" ")
    return "%s %s %s" % (a[1], a[0], a[2])


if __name__ == '__main__':
    MARKET = "gas"
    METRICS = ["price", "residual"]

    DATASET_LEN = 10
    EPOCHS = 600
    BATCH = 64
    SCALAR = 1000

    TRAIN = 0

    data_frame = pd.read_csv("data/raw.csv")

    # manipulate data frame
    mon_to_num = {
        "gen": "01", "feb": "02", "mar": "03", "apr": "04", "mag": "05", "giu": "06",
        "lug": "07", "ago": "08", "set": "09", "ott": "10", "nov": "11", "dic": "12"
    }
    data_frame['Date'] = data_frame['Date'].replace(mon_to_num, regex=True)
    data_frame["Date"] = data_frame["Date"].str.replace(',', '')
    data_frame['Date'] = data_frame['Date'].apply(reformat_date)
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%d %m %Y')

    # fix price format
    data_frame["Price"] = data_frame["Price"].str.replace(',', '.')
    data_frame["Price"] = data_frame["Price"].astype(float)

    # complete dataframe with NaN where vale is missing
    data_frame = data_frame.set_index('Date')
    date_complete = pd.date_range(start=pd.to_datetime("2017-01-01"), end=pd.to_datetime("2023-06-30"), freq='D')
    data_frame = data_frame.reindex(date_complete)

    data_frame = np.log(data_frame)
    filled_data_frame = data_frame.copy()

    # replace negative values with NaN and then remove NaN values
    data_frame = data_frame.mask(data_frame["Price"] <= 0, np.nan)
    data_frame = data_frame.dropna()

    log_prices = data_frame["Price"].values
    log_rets = differences(log_prices)

    data_frame = pd.DataFrame({"price": log_prices[1:], "residual":  log_rets})
    x_train, y_train = create_single_series_dataset(np.exp(data_frame), METRICS, DATASET_LEN, SCALAR, "residual")

    neural = Neural("refill_%s" % MARKET, METRICS, DATASET_LEN, n_output=1)
    if TRAIN == 1:
        neural.train(x_train, y_train, epochs=EPOCHS, batch=BATCH)
    elif TRAIN == 2:
        neural.resume_train(x_train, y_train, epochs=EPOCHS, batch=BATCH)
    else:
        neural.load()

    # predict NaN values
    refilled_log_prices = []
    log_prices = []
    index_to_remove = [i for i in range(len(METRICS))]

    array = np.array([])
    # for log_price in np.log(filled_data_frame["Price"].values):
    for day, row in filled_data_frame.iterrows():
        log_price = row[0]
        # check if values is not Nan or not 0 also check if values is not inf beacuase log of 0.0 is inf
        if not pd.isnull(log_price) and log_price != 0 and not np.isinf(log_price):
            log_prices.append(log_price)
            refilled_log_prices.append(log_price)

            # first step append price
            if len(array) == 0:
                array = np.append(array, log_price)
            # second iteration caluculate first log-ret, append price and log-ret and remove first price
            # do this because at first step i cannot have log-ret
            elif len(array) == 1:
                log_ret = array[0] - log_price
                array = np.delete(array, [0])
                array = np.append(array, log_price)
                array = np.append(array, log_ret)
            # all every steps
            else:
                last_log_price = array[-2]
                array = np.append(array, log_price)
                array = np.append(array, last_log_price-log_price)
    
        # find NaN value
        else:
            if len(array) == len(METRICS) * DATASET_LEN:
                # predict next log ret
               
                prediction = np.argmax(neural.predict(np.exp(array))[0])
                predicted_log_ret = np.log(prediction / SCALAR)

                # takes the last price and with predicted log ret calculates the price (calculated_price)
                last_price = array[-2]
                calculated_price = last_price + predicted_log_ret
                array = np.append(array, calculated_price)
                array = np.append(array, predicted_log_ret)

                # insert price
                refilled_log_prices.append(calculated_price)

        if len(array) > len(METRICS) * DATASET_LEN:
            array = np.delete(array, index_to_remove)


    print(len(log_prices))
    print(len(refilled_log_prices))

    print(get_results(log_rets))
    print(get_results(differences(refilled_log_prices)))

    # write csv
    # use fmt to not write numbers with scientific notation
    np.savetxt('data/original.csv', log_prices, delimiter=",", fmt='%f')
    np.savetxt('data/filled.csv', refilled_log_prices, delimiter=",", fmt='%f')

    # graph
    graph = Graph(mode="png")
    graph.refilled([log_prices, refilled_log_prices], "refill")