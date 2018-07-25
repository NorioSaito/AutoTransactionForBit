import tensorflow as tf
import inputData as id
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

#入力データを作成
input_data, t = id.input_csv_byPandas()

#入力データの列数
length_of_sequence = input_data.shape[1]
#出力層のユニット数
in_out_neurons = 3
#隠れ層の数
n_hidden = 1

#ニューラルネットワークのモデル構築
model = Sequential()
model.add(LSTM(length_of_sequence,return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
optimizer = Adam(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=optimizer)

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
#model.fit()の第一引数はNumpy配列のリストを渡すっぽい...
#input_dataはDataFrameのためエラー発生している？
model.fit(input_data, t,
          batch_size=15,
          epochs=100,
          validation_split=0.1,
          callbacks=[early_stopping]
          )

predicted = model.predict(input_data)
