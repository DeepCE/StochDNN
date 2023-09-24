import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from lib import differences, model_to_generate_trajectory


def reformat_date(value):
    a = value.split(" ")
    return "%s %s %s" % (a[1], a[0], a[2])


def setlabel(ax, label, loc=2, borderpad=0.6, fontsize=12, **kwargs):
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN,np.NaN,color='none',label=label)
    label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,fontsize=fontsize,**kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()



if __name__ == '__main__':
    '''
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
    data_frame = data_frame.mask(data_frame["Price"] <= 0, np.nan)
    data_frame = data_frame.dropna()
    original_without_nan = np.log(data_frame["Price"].values)

    original = pd.read_csv("data/original.csv").values
    filled = pd.read_csv("data/filled.csv").values
    detrend = pd.read_csv("data/detrend.csv").values[1:]

    # combine original and filled
    new_original = []
    new_filled = []
    i = 0
    j = 0
    while i != len(original):
        new_filled.append(round(filled[j][0], 3))
        if round(filled[j][0], 3) == round(original[i][0], 3):
            new_original.append(round(filled[j][0], 3))
            i += 1
            j += 1
        else:
            new_original.append(np.nan)
            j += 1

    plt.rcParams["figure.figsize"] = (14, 10)
    fig, axs = plt.subplots(4)

    fixed_locator_1 = plt.FixedLocator([x for x in range(0, len(original_without_nan), int(len(original_without_nan)/8)+50)])
    fixed_formatter_1 = plt.FixedFormatter(["2017", "2018", "2019", "2020", "2021", "2022", "2023"])

    fixed_locator = plt.FixedLocator([0, 365, 730, 1096, 1460, 1825, 2190, 2555])
    fixed_formatter = plt.FixedFormatter(["2017", "2018", "2019", "2020", "2021", "2022", "2023"])
    
    axs[0].plot(original_without_nan, linewidth=1)
    axs[0].xaxis.set_major_locator(fixed_locator_1)
    axs[0].xaxis.set_major_formatter(fixed_formatter_1)

    axs[1].plot(new_filled, linewidth=1, color="C1")
    axs[1].plot(new_original, linewidth=1, color="C0")
    axs[1].plot(filled-detrend, linewidth=2, color="C2")
    axs[1].xaxis.set_major_locator(fixed_locator)
    axs[1].xaxis.set_major_formatter(fixed_formatter)

    axs[2].plot(detrend, linewidth=1)
    axs[2].xaxis.set_major_locator(fixed_locator)
    axs[2].xaxis.set_major_formatter(fixed_formatter)

    axs[3].plot(differences(detrend), linewidth=1)
    axs[3].xaxis.set_major_locator(fixed_locator)
    axs[3].xaxis.set_major_formatter(fixed_formatter)

    setlabel(axs[0], '(a)')
    setlabel(axs[1], '(b)')
    setlabel(axs[2], '(c)')
    setlabel(axs[3], '(d)')

    plt.savefig("graph/info.png")
    plt.savefig("graph/info.eps", format='eps')
     '''

    '''
    AI TRAINING GRAPH
    
    with open("models/sim_gas-result.json") as file:
        data = json.load(file)
        sim_loss = list(data["loss"].values())

    plt.rcParams["figure.figsize"] = (14, 5)
    fig, axs = plt.subplots()

    axs.plot(sim_loss, linewidth=1)

    for tick in axs.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    for tick in axs.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    plt.savefig("graph/training.png")
    plt.savefig("graph/training.eps", format='eps')
    '''

    '''
    AI REBUILD
    '''

    END_DATE = "2023-06-30"

    data_frame = pd.read_csv("data/detrend.csv")
    log_prices = data_frame["Price"].values
    log_ret = differences(log_prices)

    date_list = pd.date_range(end=END_DATE, periods=len(log_ret), freq='D')
    df = pd.DataFrame({"date": date_list, "prices": log_prices[1:], "residual":  log_ret})
    df = df[df["date"] >= "2017-07-01"]

    WINDOW = 365
    test_df = df[len(df)-WINDOW-365:len(df)-365]
    test_log_prices_1 = test_df["prices"].values
    #AI
    gen_df = pd.read_csv('data/in_of_sample_15_%s.csv' % WINDOW)
    gen_df = gen_df.transpose()
    ai_matrix_1 = gen_df.values.tolist()
    #LL
    model_ll_matrix_1 = model_to_generate_trajectory("ll", 1, WINDOW, test_log_prices_1[0])
    #MM
    model_mm_matrix_1 = model_to_generate_trajectory("mm", 1, WINDOW, test_log_prices_1[0])

    WINDOW = 365
    test_df = df[len(df)-WINDOW:]
    test_log_prices_2 = test_df["prices"].values
    #AI
    gen_df = pd.read_csv('data/out_of_sample_15_%s.csv' % WINDOW)
    gen_df = gen_df.transpose()
    ai_matrix_2 = gen_df.values.tolist()
    #LL
    model_ll_matrix_2 = model_to_generate_trajectory("ll", 1, WINDOW, test_log_prices_2[0])
    #MM
    model_mm_matrix_2 = model_to_generate_trajectory("mm", 1, WINDOW, test_log_prices_2[0])

    WINDOW = 1826
    test_df = df[len(df)-WINDOW:]
    test_log_prices_3 = test_df["prices"].values
    #AI
    gen_df = pd.read_csv('data/out_of_sample_15_%s.csv' % WINDOW)
    gen_df = gen_df.transpose()
    ai_matrix_3 = gen_df.values.tolist()
    #LL
    model_ll_matrix_3 = model_to_generate_trajectory("ll", 1, WINDOW, test_log_prices_3[0])
    #MM
    model_mm_matrix_3 = model_to_generate_trajectory("mm", 1, WINDOW, test_log_prices_3[0])

    #AI
    plt.rcParams["figure.figsize"] = (14, 10)
    fig, axs = plt.subplots(3)
    axs[0].plot(test_log_prices_1, linewidth=1)
    axs[0].plot(ai_matrix_1[2], linewidth=1)
    axs[1].plot(test_log_prices_2, linewidth=1)
    axs[1].plot(ai_matrix_2[17], linewidth=1)
    axs[2].plot(test_log_prices_3, linewidth=1)
    axs[2].plot(ai_matrix_3[13], linewidth=1)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[2].tick_params(axis='both', labelsize=12)
    setlabel(axs[0], '(a)')
    setlabel(axs[1], '(b)')
    setlabel(axs[2], '(c)')
    plt.savefig("ai_graph/ai_series.png")
    plt.savefig("ai_graph/ai_series.eps", format='eps')

    #LL
    plt.rcParams["figure.figsize"] = (14, 10)
    fig, axs = plt.subplots(3)
    axs[0].plot(test_log_prices_1, linewidth=1)
    axs[0].plot(model_ll_matrix_1[0], linewidth=1)
    axs[1].plot(test_log_prices_2, linewidth=1)
    axs[1].plot(model_ll_matrix_2[0], linewidth=1)
    axs[2].plot(test_log_prices_3, linewidth=1)
    axs[2].plot(model_ll_matrix_3[0], linewidth=1)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[2].tick_params(axis='both', labelsize=12)
    setlabel(axs[0], '(a)')
    setlabel(axs[1], '(b)')
    setlabel(axs[2], '(c)')
    plt.savefig("ai_graph/ll_series.png")
    plt.savefig("ai_graph/ll_series.eps", format='eps')

    #MM
    plt.rcParams["figure.figsize"] = (14, 10)
    fig, axs = plt.subplots(3)
    axs[0].plot(test_log_prices_1, linewidth=1)
    axs[0].plot(model_mm_matrix_1[0], linewidth=1)
    axs[1].plot(test_log_prices_2, linewidth=1)
    axs[1].plot(model_mm_matrix_2[0], linewidth=1)
    axs[2].plot(test_log_prices_3, linewidth=1)
    axs[2].plot(model_mm_matrix_3[0], linewidth=1)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[2].tick_params(axis='both', labelsize=12)
    setlabel(axs[0], '(a)')
    setlabel(axs[1], '(b)')
    setlabel(axs[2], '(c)')
    plt.savefig("ai_graph/mm_series.png")
    plt.savefig("ai_graph/mm_series.eps", format='eps')