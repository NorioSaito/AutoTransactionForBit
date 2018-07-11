import csv
import pandas as pd

#CSVファイルをListとして読み込む
#データ15件の最初と最後の取引価格の差額を算出
def input_csv():
	csv_file = open("C:/Users/nsait/AutoTransactionForBit/data_201806701.csv", "r", encoding="ms932", errors="", newline="" )
	#リスト形式
	data_list = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
	#リストを1行ずつ抽出
	count = 0
	ltp_start = 0
	ltp_end = 0
	for row in data_list:
		if count == 0:
			ltp_start = float(row[1])
		elif count == 14:
			ltp_end = float(row[1])

			if ltp_start > ltp_end:
				print('-', ltp_start - ltp_end)
			elif ltp_end >= ltp_start:
				print('+', ltp_end - ltp_start)
				count = 0
		count += 1
		print(row)

#CSVファイルをPandasで取得
def input_csv_byPandas():
	data_list = pd.read_csv('C:/Users/nsait/AutoTransactionForBit/data_201806701.csv', header=None)
	print(data_list)

input_csv_byPandas()
