import csv

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

input_csv()
