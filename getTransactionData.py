'''
Created on 2018/06/03

@author: user
1分ごとに取引データを取得し、グラフ描画するクラス
'''
# グラフ描画に使う
import numpy as np
import matplotlib.pyplot as plt
# bitFlyerにアクセスするのに使う
import pybitflyer
# 時間の管理に使う
import time
from datetime import datetime
# リアルタイムプロットに必要
from IPython.display import display, clear_output
# Jupyter環境でグラフを表示するのに必要
#%matplotlib inline

api = pybitflyer.API()
ticker = api.ticker(product_code = "BTC_JPY")

print(ticker)

# 最終取引価格、移動平均、標準偏差を格納する配列
raws = []
sma1, sma2 = [], []
sgm1, sgm2 = [], []

# 移動平均を取る幅
itr1 = 15 # 15 mins
itr2 = 60  # 60 mins

# 60分間の最終取引価格の配列
current_price = api.ticker(product_code = "BTC_JPY")['ltp']
ltps2 = current_price*np.ones(itr2)

plt.ion()
fig = plt.figure(figsize=(16,5))
axe = fig.add_subplot(111)

while True:
    # 60秒ごとに稼働
    if datetime.now().strftime('%S') [1:2]== '0':
        clear_output(wait = True)
        tick = api.ticker(product_code = "BTC_JPY")
        # 最終取引価格の更新
        ltps2 = np.hstack((ltps2[1:itr2], tick['ltp']))
        ltps1 = ltps2[itr2-itr1:itr2]
        # プロット用データの更新
        raws = np.append(raws, [ltps1[itr1-1]])
        sma1 = np.append(sma1, [ltps1.mean()])
        sma2.append(ltps2.mean())
        sgm1 = np.append(sgm1, [ltps1.std()])
        sgm2 = np.append(sgm2, [ltps2.std()])
        # プロット
        axe.plot(raws, "black", linewidth=5, label="Raw price")
        axe.plot(sma1, "r", linewidth=5, label="15min SMA")
        axe.plot(sma2, "g", linewidth=5, label="60min SMA")
        axe.plot(sma1+2*sgm1, "r", linewidth=1, linestyle="dashed", label="15min 2sigma")
        axe.plot(sma1-2*sgm1, "r", linewidth=1, linestyle="dashed")
        axe.plot(sma2+2*sgm2, "g", linewidth=1, linestyle="dashed", label="60min 2sigma")
        axe.plot(sma2-2*sgm2, "g", linewidth=1, linestyle="dashed")
        axe.legend(loc='upper left')
        axe.set_title("SMA and Bollinger band")
        display(fig)
        # 次の00秒まで休憩
        time.sleep(6)
        axe.cla()