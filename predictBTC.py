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
from matplotlib import pyplot as plt

#価格データの取得
polo = poloniex.Poloniex()
polo.timeout = 2
rawdata = polo.returnChartData('USDT_BTC',
                               period=300,
                               start=time.time()-polo.DAY*180,
                               end=time.time())
print(rawdata)
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

#ニューラルネットワークモデルの作成
in_out_neurons = 1
hidden_neurons = 300
length_of_sequences = 50

model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam",)

#学習の実施
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
history = model.fit(X_train, y_train, batch_size=600, epochs=10, validation_split=0.1, callbacks=[early_stopping])

json_string = model.to_json()
open('keras_lstm_model.json', 'w').write(json_string)

model.save_weights('keras_lstm_weihgts.h5')

#グラフ描画
pred_data = model.predict(X_train)
plt.plot(y_train, label='train')
plt.plot(pred_data, label='pred')
plt.legend(loc='upper left')
plt.show()
