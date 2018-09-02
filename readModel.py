import poloniex
import time
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import model_from_json
from matplotlib import pyplot as plt


def createDate():
    #価格データの取得
    input_data = pd.read_csv('data_of_pred.csv', header=None)
    #データの個数調整
    ##input_data = input_data[len(input_data) % 15:]
    #t = t[len(t) % 15:]
    price_data = pd.DataFrame(input_data)
    mss = MinMaxScaler()
    input_dataFrame = pd.DataFrame(mss.fit_transform(price_data))
    dataframe = input_dataFrame.iloc[0:]
    docX = []
    for i in range(len(dataframe)):
        docX.append(dataframe.iloc[i:].values)
    alsX = np.array(docX)
    return alsX
#保存したモデルを読み込み
def model_read(input_data):
    json_string = open('keras_lstm_model.json').read()
    m = model_from_json(json_string)

    m.summary()
    m.compile(loss="mean_squared_error", optimizer="adam",)
    m.load_weights('keras_lstm_weihgts.h5')

    pred_data = m.predict(input_data)
    plt.plot(input_data[:,0], label='train')
    plt.plot(pred_data, label='pred')
    plt.legend(loc='upper left')
    plt.show()
input_array = createDate()
model_read(input_array)
