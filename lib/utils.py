from scipy import stats
import numpy as np
from scipy.stats import wasserstein_distance


def get_results(obs):
    kt = stats.kurtosis(obs)
    st = np.std(obs)
    sk = stats.skew(obs)
    mn = np.mean(obs)
    return kt + 3, st, sk, mn


def differences(obs):
    dif = []
    for i in range(len(obs)):
        if i != 0:
            dif.append(obs[i] - obs[i-1])
    return dif


def wasserstein(matrix, original):
    array = []
    for series in matrix:
        array.append(wasserstein_distance(differences(series), original))
    return np.mean(array)


class PatternReconigzer():
    def __init__(self, max_repetitions=60):
        self.max_repetitions = max_repetitions

    def recognize(self, array):
        dic = {}
        i = 0
        while i < len(array):
            if array[i] in array[i + 1:]:
                index = array[i + 1:].index(array[i])
                length = self.seq_len(array[i:], array[i + index + 1:])
                i += length
                if length not in dic:
                    dic[length] = 0
                dic[length] += 1
                continue
            i += 1
        return self.check_loop(dic)

    @staticmethod
    def seq_len(a1, a2):
        i = 0
        for e1, e2 in zip(a1, a2):
            if e1 != e2:
                break
            i += 1
        return i

    def check_loop(self, dic):
        for key in dic:
            if key >= self.max_repetitions:
                return True
        return False
