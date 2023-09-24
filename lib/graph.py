import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class Graph:
    def __init__(self, path, mode="all"):
        self.path = path
        self.mode = mode

    def save(self, name):
        if self.mode in ["png", "all"]:
            plt.savefig(self.path % name)
        if self.mode in ["eps", "all"]:
            plt.savefig(self.path % name, format='eps')

    def refilled(self, multiple_array, name):
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(len(multiple_array))
        for i in range(len(multiple_array)):
            axs[i].plot(multiple_array[i], linewidth=1)
        self.save(name)

    def rebuild(self, multiple_array, original, name):
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(len(multiple_array))
        for i in range(len(multiple_array)):
            axs[i].plot(original, linewidth=1)
            axs[i].plot(multiple_array[i], linewidth=1)
        self.save(name)

    def gen_ai(self, original, ai, name):
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(1)
        axs.plot(original, linewidth=1)
        axs.plot(ai, linewidth=1)
        self.save(name)

    def generated(self, array, original, name):
        title_dic = {0: "AI", 1: "LL", 2: "MM"}
        color_dic = {0: "C1", 1: "C2", 2: "C3"}
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(len(array))
        for i in range(len(array)):
            axs[i].plot(original, linewidth=1)
            axs[i].plot(array[i], linewidth=1, color=color_dic[i])
            axs[i].title.set_text(title_dic[i])
        self.save(name)

    def gen_bis(self, array, original, name):
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(len(array))
        for i in range(len(array)):
            axs[i].plot(original, linewidth=1)
            axs[i].plot(array[i], linewidth=1)
        self.save(name)

    def errors(self, dic, name, labels):
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(2)

        step_size = len(labels) // 5 
        reduced_labels = labels[::step_size]

        reduced_locator = plt.FixedLocator([i for i in range(0, len(labels), step_size)])
        reduced_formatter = plt.FixedFormatter(reduced_labels)
        
        axs[0].plot(dic["mae"]["ai"], linewidth=1)
        axs[0].plot(dic["mae"]["ll"], linewidth=1)
        axs[0].plot(dic["mae"]["mm"], linewidth=1)
        axs[0].xaxis.set_major_locator(reduced_locator)
        axs[0].xaxis.set_major_formatter(reduced_formatter)
        axs[0].legend(['DNN', 'M1', 'M2'])
        axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[0].tick_params(axis='both', labelsize=14)

        axs[1].plot(dic["rmse2"]["ai"], linewidth=1)
        axs[1].plot(dic["rmse2"]["ll"], linewidth=1)
        axs[1].plot(dic["rmse2"]["mm"], linewidth=1)
        axs[1].xaxis.set_major_locator(reduced_locator)
        axs[1].xaxis.set_major_formatter(reduced_formatter)
        axs[1].legend(['DNN', 'M1', 'M2'])
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[1].tick_params(axis='both', labelsize=14)

        self.save(name)
