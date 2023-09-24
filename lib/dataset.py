import numpy as np
import pandas as pd



def create_single_series_dataset(data_frame, metrics, dataset_len, scalar, main_metric):
    m_index = metrics.index(main_metric)
    n_metrics = len(metrics)
    index_to_remove = np.arange(0, n_metrics)

    x_train = np.empty((0, n_metrics*dataset_len), int)
    y_train = np.array([])
    array = np.array([])

    for index, row in data_frame.iterrows():
        if len(array) == n_metrics*dataset_len:
            x_train = np.append(x_train, array.reshape(1, len(array)), axis=0)
            y_train = np.append(y_train, row[m_index] * scalar)
            array = np.delete(array, index_to_remove)
        array = np.append(array, row)

    return x_train, y_train


def create_series_dataset(data_frame, metrics, dataset_len, scalar):
    n_metrics = len(metrics)
    array = np.array([])
    x_train = np.empty((0, n_metrics*dataset_len), int)
    index_to_remove = np.arange(0, n_metrics)

    y_df = pd.DataFrame(columns=metrics)
    for index, row in data_frame.iterrows():
        if not pd.isnull(row[0]):
            if len(array) == n_metrics*dataset_len:
                x_train = np.append(x_train, array.reshape(1, len(array)), axis=0)
                y_df.loc[len(y_df)] = row
                array = np.delete(array, index_to_remove)
            array = np.append(array, row)
            
    y_train = []
    for metric in metrics:
        y = y_df[[metric]].values * scalar
        y = y.astype(int)
        y_train.append(y)
    return x_train, y_train