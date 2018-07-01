# グラフ描画に使う
import numpy as np
# bitFlyerにアクセスするのに使う
import pybitflyer
# 時間の管理に使う
import time
from datetime import datetime
# リアルタイムプロットに必要
# Jupyter環境でグラフを表示するのに必要
#%matplotlib inline

#d売り買いの閾値
th1 = 1/56000
th2 = 1/8000

    # 指値買い注文
def buy_btc_lmt(amt, btc, jpy, prc):
    amt = int(amt*100000000)/100000000
    #buy = api.sendchildorder(product_code="BTC_JPY", child_order_type="LIMIT", price=prc, side="BUY", size=amt, minute_to_expire=10, time_in_force="GTC")
    btc += amt
    jpy -= amt * prc
    return btc * prc + jpy

# 指値売り注文
def sell_btc_lmt(amt, btc, jpy, prc):
    amt = int(amt*100000000)/100000000
    #sell = api.sendchildorder(product_code="BTC_JPY", child_order_type="LIMIT", price=prc, side="SELL", size=amt, minute_to_expire=10, time_in_force="GTC")
    btc -= amt
    jpy += amt * prc
    return btc * prc + jpy

def compute(x, th1, th2):
    if 0 < x and x < th1:
        out = 0
    elif th1 <= x and x <= th2:
        out = -1/(th1-th2)**2*(x-th2)**2+1
    elif th2 < x:
        out = 1
    else:
        out = 0
    return out
api = pybitflyer.API()
# 移動平均を取る幅[分]
itr = 20
# 最終取引価格と移動平均を一時的に保存する配列を用意
current_price = api.ticker(product_code = "BTC_JPY")['ltp']
ltps = current_price*np.ones(itr)
smas = current_price*np.ones(2)
# 最小取引額[BTC]
min_btc = 0.001
# 資産と時間を格納する配列。あとで確認するときに使える
jpys = []
btcs = []
tms = []
# 最終取引価格と移動平均を格納する配列。あとで確認するときに使える
raw = []
smoothed = []

#日本円
balanceJpy = 5000000
#BTC
balanceBtc = 0
while True:
# 00秒に稼働
    if datetime.now().strftime('%S') [0:2]== '00':
        # 資産を取得し格納
        #balance = api.getbalance()
        #jpy = balance[0]['available']
        jpy = balanceJpy
        jpys.append(jpy)
        #btc = balance[1]['available']
        btc = balanceBtc
        btcs.append(btc)
        #時間を文字列で取得し格納
        tick = api.ticker(product_code = "BTC_JPY")
        tm = tick['timestamp']
        tm = str((int(tm[11:13])+9)%24) + tm[13:19]
        tms.append(tm)
        # 最終取引価格と移動平均の更新
        ltps = np.hstack((ltps[1:itr], tick['ltp']))
        smas = np.hstack((smas[1], ltps.mean()))
        # 確認用データの更新
        raw.append(ltps[itr-1])
        smoothed.append(smas[1])
        # 移動平均の分率利率
        r = (smas[1]-smas[0])/smas[0]

        # 利率が正の時はBTC買い
        if r > 0:
            # JPY資産のうちどれだけBTCに変えるかを計算
            amt_jpy = compute(r, th1, th2)*jpy
            amt_btc = amt_jpy/ltps[itr-1]
            # 購入量が最小取引額を超えていれば指値買い
            if amt_btc > min_btc:
                #print(buy_btc_lmt(amt_btc, balanceBtc, balanceJpy, current_price))
                balanceJpy = balanceJpy - (amt_btc * current_price)
                balanceBtc = balanceBtc + amt_btc
            else:
                print("取引なし")
        # 利率が負の時はBTC売り
        if r < 0:# BTC資産のうちどれだけJPYに変えるかを計算
            amt_btc = compute(-r, th1, th2)*btc
            # 売却量が最小取引額を超えていれば指値売り
            if amt_btc > min_btc:
                print(sell_btc_lmt(amt_btc, balanceBtc, balanceJpy, current_price))
                balanceJpy = balanceJpy + (amt_btc * current_price)
                balanceBtc = balanceBtc - amt_btc
                print("jpy = ", balanceJpy)
                print("btc = ", balanceBtc * current_price)
            # 次の00秒まで休憩
        time.sleep(57)