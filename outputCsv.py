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
#from asn1crypto._ffi import null
# リアルタイムプロットに必要
# Jupyter環境でグラフを表示するのに必要
#%matplotlib inline

#csv出力処理
def outputCsv(tick):
    f = open('data_201806701.csv', 'a')
    writer = csv.writer(f)

# データをリストに保持
    csvlist = []
    print(tick['timestamp'])
    print(tick['ltp'])
    csvlist.append(tick['timestamp'])
    csvlist.append(tick['ltp'])
    csvlist.append(tick['best_ask'])
    csvlist.append(tick['best_bid'])
    csvlist.append(tick['best_ask_size'])
    csvlist.append(tick['best_bid_size'])
    csvlist.append(tick['total_ask_depth'])
    csvlist.append(tick['total_bid_depth'])
    csvlist.append(tick['volume_by_product'])

# 出力
    writer.writerow(csvlist)

# ファイルクローズ
    f.close()

api = pybitflyer.API()

while True:
# 00秒に稼働
    if datetime.now().strftime('%S') [0:2]== '00':

        tick = api.ticker(product_code = "BTC_JPY")

        outputCsv(tick)
        time.sleep(57)