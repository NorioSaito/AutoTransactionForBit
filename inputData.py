import csv

csv_file = open("C:/Users/nsait/AutoTransactionForBit/data_201806701.csv", "r", encoding="ms932", errors="", newline="" )
#リスト形式
data_list = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
#リストを1行ずつ抽出
for row in data_list:
	print(row)
