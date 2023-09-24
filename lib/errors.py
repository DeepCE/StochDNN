import numpy as np


def compute_errors(original_series, matrix, mode):
    errors = []
    for gen_series in matrix:
        error_array = []
        for original, predicted in zip (original_series, gen_series):
            if mode == "mae":
                error_array.append(abs(original - predicted))
            else:
                error_array.append((original - predicted)**2)
        errors.append(error_array)
    return errors
