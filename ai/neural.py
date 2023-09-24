from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_limit):
        self.loss_limit = loss_limit

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss = logs.get('loss')
        if loss < self.loss_limit:
            print("\n\nTraining stopped as loss is less than 0.1")
            self.model.stop_training = True


class Neural(object):
    def __init__(self, market, metrics, dataset_len, n_output=1):
        self.metrics = metrics
        self.n_metrics = len(metrics)
        self.dataset_len = dataset_len
        self.n_output = n_output
        self.model = None
        self.path = "models/%s" % market

    def train(self, x_train, y_train, epochs, batch, loss_limit=0, x_test=None, y_test=None):
        embedding_dim = 10 * self.dataset_len
        input_layer = layers.Input(shape=(x_train.shape[1],))
        x = layers.Embedding(40000, embedding_dim, input_length=self.n_metrics * 10)(input_layer)
        x = layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(256, dropout=0.2, return_sequences=True)(x)
        x = layers.LSTM(128, dropout=0.1)(x)

        if self.n_output == 1:
            out = layers.Dense(max(y_train) + 2, activation="softmax")(x)
        else:
            out = []
            for i in range(self.n_output):
                out.append(layers.Dense(max(y_train[i]) + 2, activation="softmax")(x))

        self.model = Model(inputs=input_layer, outputs=out)

        self.model.summary()
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        callback = CustomCallback(loss_limit)
        result = self.model.fit(x_train, y_train, batch_size=batch, epochs=epochs, callbacks=[callback])
        if x_test is not None and y_test is not None:
            self.model.evaluate(x_test, y_test)
        self.save_model(result)

    def resume_train(self, x_train, y_train, epochs, batch, loss_limit=0):
        self.load()
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        callback = CustomCallback(loss_limit)
        result = self.model.fit(x_train, y_train, batch_size=batch, epochs=epochs, callbacks=[callback])
        self.save_model(result)

    def save_model(self, result):
        with open(self.path + ".json", "w+") as file:
            file.write(self.model.to_json())
        self.model.save_weights(self.path + ".h5")

        with open(self.path + "-result.json", "w+") as file:
            result = pd.DataFrame(result.history)
            result.to_json(file)

    def load(self, schema=True):
        with open(self.path + ".json", "r") as file:
            loaded_model_json = file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.path + ".h5")
        if schema:
            self.model.summary()

    def predict(self, array, length_array=None):
        if length_array is None:
            length_array = self.n_metrics * self.dataset_len

        num_array = np.asarray(array)
        num_array = num_array.reshape((1, length_array))
        return self.model.predict(num_array, verbose=0)
