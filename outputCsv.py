'''
#1分ごとに取引データを取得し、CSV出力するクラス
'''
# bitFlyerにアクセスするのに使う
import pybitflyer
# 時間の管理に使う
import time
from datetime import datetime
#import sys
import csv
import readModel as rm
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import model_from_json
from matplotlib import pyplot as plt

#from asn1crypto._ffi import null
# リアルタイムプロットに必要
# Jupyter環境でグラフを表示するのに必要
#%matplotlib inline

#csv出力処理
def outputCsv(tick):
    f = open('data_of_pred.csv', 'a')
    writer = csv.writer(f)

# データをリストに保持
    csvlist = []
    print(tick['timestamp'])
    print(tick['ltp'])
    #取得日時をデータに入れると標準偏差を求める時にエラーが起きるのでコメントアウト
    #csvlist.append(tick['timestamp'])#取得日時
    csvlist.append(tick['ltp'])#最終取引価格
    csvlist.append(tick['best_ask'])#最高買い価格
    csvlist.append(tick['best_bid'])#最低売り価格
    csvlist.append(tick['best_ask_size'])#最高買い価格の数
    csvlist.append(tick['best_bid_size'])#最低売り価格の数
    csvlist.append(tick['total_ask_depth'])#買い注文総数
    csvlist.append(tick['total_bid_depth'])#売り注文総数
    csvlist.append(tick['volume_by_product'])#価格ごとの出来高

# 出力
    writer.writerow(csvlist)

# ファイルクローズ
    f.close()

api = pybitflyer.API()
count = 0
while True:
# 00秒に稼働
    if datetime.now().strftime('%S') [0:2]== '00':
        print(count)
        tick = api.ticker(product_code = "BTC_JPY")

        if count == 15:
            outputCsv(tick)
            input_data = pd.read_csv('data_of_pred.csv', header=None)
            rm.model_read(input_data)
            count = 0
        else:
            outputCsv(tick)
            count = count + 1
        time.sleep(57)
