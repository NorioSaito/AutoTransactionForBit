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



#価格データの取得
polo = poloniex.Poloniex()
polo.timeout = 2
rawdata = polo.returnChartData('USDT_BTC',
                               period=300,
                               start=time.time()-polo.DAY*10,
                               end=time.time())

#データの前処理
price_data = pd.DataFrame([float(i.get('open')) for i in rawdata])
mss = MinMaxScaler()
input_dataframe = pd.DataFrame(mss.fit_transform(price_data))

#訓練データと検証データの分割
def _load_data(data, n_prev=50):
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev=50):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = train_test_split(input_dataframe)

#保存したモデルを読み込み
json_string = open('keras_lstm_model.json').read()
m = model_from_json(json_string)

m.summary()
m.compile(loss="mean_squared_error", optimizer="adam",)
m.load_weights('keras_lstm_weihgts.h5')

pred_data = m.predict(X_train)
plt.plot(y_train, label='train')
plt.plot(pred_data, label='pred')
plt.legend(loc='upper left')
plt.show()
