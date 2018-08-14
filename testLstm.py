import pandas as pd
import tensorflow as tf
import inputData as id
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

BATCH_SIZE  = 50
HIDDEN_SIZE = 150
NB_EPOCH    = 100

def generate_sine_data():
    STEPS_PAR_CYCLE  = 50
    NUMBER_OF_CYCLES = 100

    df = pd.DataFrame(np.arange(STEPS_PAR_CYCLE * NUMBER_OF_CYCLES + 1), columns= ['x'])
    df['sine_x'] =  df.x.apply(lambda x: math.sin(x * (2 * math.pi / STEPS_PAR_CYCLE)))
    return df['sine_x']

dataframe = generate_sine_data()
dataset   = dataframe.values.astype('float32')

scaler  = MinMaxScaler(feature_range=(0, 1) )
dataset = scaler.fit_transform(dataset)

LENGTH      = len(dataset)
train_size  = int(LENGTH * 0.67)
test_size   = LENGTH - train_size
train, test = dataset[:train_size], dataset[train_size:]

def create_dataset(dataset, time_steps= 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i : (i + time_steps)]
        dataX.append(a)
        dataY.append(dataset[i + time_steps])
    return np.array(dataX), np.array(dataY)

TIME_STEPS = 3
trainX, trainY = create_dataset(train, TIME_STEPS)
testX,  testY  = create_dataset(test,  TIME_STEPS)

trainX = trainX[ len(trainX) % batch_size: ]
trainY = trainY[ len(trainY) % batch_size: ]
testX = testX[ len(testX) % batch_size: ]
testY = testY[ len(testY) % batch_size: ]

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX  = np.reshape(testX,  (testX.shape[0],  testX.shape[1],  1))

model = Sequential()
model.add( LSTM(HIDDEN_SIZE, batch_input_shape=(BATCH_SIZE, TIME_STEPS, 1)) )
model.add( Dense(1) )
model.compile(loss= 'mean_squared_error', optimizer= 'adam')

#x : 直前の層の出力
#W : shape= [直前の層の出力数、Dense()の出力数] の重み
#b : shape= [Dense() の出力数] のバイアス

#Dense() の出力 = x・W + b   # ・は行列の積

model.fit(trainX, trainY, nb_epoch= NB_EPOCH, batch_size= BATCH_SIZE)
