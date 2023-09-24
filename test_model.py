from lib import model_to_generate_trajectory, differences, Graph
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def setlabel(ax, label, loc=2, borderpad=0.6, fontsize=12, **kwargs):
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN,np.NaN,color='none',label=label)
    label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,fontsize=fontsize,**kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()



PREFIX = "in"
WINDOW = 365
REGEN = False


if __name__ == '__main__':
    END_DATE = "2023-06-30"

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


    date_list = pd.date_range(end=END_DATE, periods=len(log_ret), freq='D')
    df = pd.DataFrame({"date": date_list, "prices": log_prices[1:], "residual":  log_ret})
    df = df[df["date"] >= "2017-07-01"]

    if REGEN:
        model_ll_matrix = model_to_generate_trajectory("ll", 2000, WINDOW, test_log_prices[0])
        gen_df = pd.DataFrame(model_ll_matrix)
        gen_df = gen_df.transpose()
        gen_df.to_csv('data_model/ll_model_%s_%s.csv' % (PREFIX, WINDOW), index=False)

        model_mm_matrix = model_to_generate_trajectory("mm", 2000, WINDOW, test_log_prices[0])
        gen_df = pd.DataFrame(model_mm_matrix)
        gen_df = gen_df.transpose()
        gen_df.to_csv('data_model/mm_model_%s_%s.csv' % (PREFIX, WINDOW), index=False)
    else:
        gen_df = pd.read_csv('data_model/ll_model_%s_%s.csv' % (PREFIX, WINDOW))
        gen_df = gen_df.transpose()
        model_ll_matrix = gen_df.values.tolist()

        gen_df = pd.read_csv('data_model/mm_model_%s_%s.csv' % (PREFIX, WINDOW))
        gen_df = gen_df.transpose()
        model_mm_matrix = gen_df.values.tolist()

    graph = Graph("data_model/%s", mode="png")
    if WINDOW == 1826:
        for matrix, sign in zip([model_ll_matrix, model_mm_matrix], ["ll", "mm"]):
            print(sign.upper())
            saved = []
            for i, array in enumerate(matrix):
                if max(array) > 1.6 and sign == "mm":
                    print("index: %s" % i)
                    saved.append(array)
                if max(array) > 1.3 and sign == "ll":
                    print("index: %s" % i)
                    saved.append(array)
        
            print("N. intrest series: %s" % len(saved))

            if len(saved) >= 3:
                graph.generated([saved[0], saved[1], saved[2]], test_log_prices, "%s_%s_%s" % (sign, PREFIX, WINDOW))
            print("*" * 30)
    else:
        # graph.generated([model_ll_matrix[0], model_ll_matrix[1], model_ll_matrix[2]], test_log_prices, "%s_%s_%s" % ("ll", PREFIX, WINDOW))
        # graph.generated([model_mm_matrix[0], model_mm_matrix[1], model_mm_matrix[2]], test_log_prices, "%s_%s_%s" % ("mm", PREFIX, WINDOW))
        print("Start")


    '''
    FINAL GRAPH
    '''
    
    # ORIGINAL

    # in
    WINDOW = 365
    train_df = df[:len(df)-WINDOW-365]
    test_df = df[len(df)-WINDOW-365:len(df)-365]
    test_log_prices_1 = test_df["prices"].values

    # out
    train_df = df[:len(df)-WINDOW]
    test_df = df[len(df)-WINDOW:]
    test_log_prices_2 = test_df["prices"].values

    WINDOW = 1826
    train_df = df[:len(df)-WINDOW]
    test_df = df[len(df)-WINDOW:]
    test_log_prices_3 = test_df["prices"].values


    # LL
    gen_df = pd.read_csv('data_model/ll_model_%s_%s.csv' % ("in", 365))
    gen_df = gen_df.transpose()
    model_ll_matrix_1 = gen_df.values.tolist()

    gen_df = pd.read_csv('data_model/ll_model_%s_%s.csv' % ("out", 365))
    gen_df = gen_df.transpose()
    model_ll_matrix_2 = gen_df.values.tolist()

    gen_df = pd.read_csv('data_model/ll_model_%s_%s.csv' % ("out", 1826))
    gen_df = gen_df.transpose()
    model_ll_matrix_3 = gen_df.values.tolist()

    plt.rcParams["figure.figsize"] = (14, 10)
    fig, axs = plt.subplots(3)
    axs[0].plot(test_log_prices_1, linewidth=1)
    axs[0].plot(model_ll_matrix_1[0], linewidth=1)
    axs[1].plot(test_log_prices_2, linewidth=1)
    axs[1].plot(model_ll_matrix_2[14], linewidth=1)
    axs[2].plot(test_log_prices_3, linewidth=1)
    axs[2].plot(model_ll_matrix_3[500], linewidth=1)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[2].tick_params(axis='both', labelsize=12)
    setlabel(axs[0], '(a)')
    setlabel(axs[1], '(b)')
    setlabel(axs[2], '(c)')
    plt.savefig("ai_graph/ll_series.png")
    plt.savefig("ai_graph/ll_series.eps", format='eps')

    #MM
    gen_df = pd.read_csv('data_model/mm_model_%s_%s.csv' % ("in", 365))
    gen_df = gen_df.transpose()
    model_mm_matrix_1 = gen_df.values.tolist()

    gen_df = pd.read_csv('data_model/mm_model_%s_%s.csv' % ("out", 365))
    gen_df = gen_df.transpose()
    model_mm_matrix_2 = gen_df.values.tolist()

    gen_df = pd.read_csv('data_model/mm_model_%s_%s.csv' % ("out", 1826))
    gen_df = gen_df.transpose()
    model_mm_matrix_3 = gen_df.values.tolist()


    plt.rcParams["figure.figsize"] = (14, 10)
    fig, axs = plt.subplots(3)
    axs[0].plot(test_log_prices_1, linewidth=1)
    axs[0].plot(model_mm_matrix_1[12], linewidth=1)
    axs[1].plot(test_log_prices_2, linewidth=1)
    axs[1].plot(model_mm_matrix_2[10], linewidth=1)
    axs[2].plot(test_log_prices_3, linewidth=1)
    axs[2].plot(model_mm_matrix_3[584], linewidth=1)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[2].tick_params(axis='both', labelsize=12)
    setlabel(axs[0], '(a)')
    setlabel(axs[1], '(b)')
    setlabel(axs[2], '(c)')
    plt.savefig("ai_graph/mm_series.png")
    plt.savefig("ai_graph/mm_series.eps", format='eps')