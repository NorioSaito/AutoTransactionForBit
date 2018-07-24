import tensorflow as tf
import numpy as np

num_of_input_nodes = 9 #入力層のノード数（日付を抜いた要素数）
num_of_hidden_nodes = 15 #中間層のノード数（とりあえず15でのちに調整する）
num_of_output_nodes = 3 #出力層のノード数(価格が「上がる」「下がる」「横這い」の3ユニット)
num_of_traning_epochs = 5000 #トレーニングの繰り返し回数(過学習が起きないよう調整する)
size_of_mini_batch = 150 #ミニバッチあたりのサンプル数(15x10で設定)
learning_rate = 0.01 #学習率(とりあえず0.01でやってみる)
forget_bias = 1 #よくわからない（デフォルトが1らしい）
num_of_sample = 7000 #データサンプル数

#入力データを作成
def create_data(data_list):
    #csv出力したデータを行列として保持
    X = data_list.values

    t = np.empty(3)
    #データ15件ごとの価格変化を算出
	count = 0
	ltp_start = 0
	ltp_end = 0
	for row in data_list.itertuples():
		if count == 0:
			#print(row[2])
			ltp_start = row[2]
			count += 1
		elif count == 14:
			#print(row[2])
			ltp_end = row[2]
			ltp_gap = ltp_end - ltp_start

            #価格変化をone-hot表現で表す
            #現時点ではなぜか[0, 0, 0]しか作れない...。
			if ltp_gap >= 500:
				np.vstack((t, [1, 0, 0]))
			elif ltp_gap <= -500:
				np.vstack((t, [0, 1, 0]))
			else:
				np.vstack((t, [0, 0, 1]))
			count = 0
		else:
			count += 1
    return X, t

#LSTMを設計する関数
def inference(input_ph, istate_ph):
    with tf.name_scope("inference") as scope:
        #重みの構造を設定
        #重み1=入力ノード数(8)と中間ノード数(15)に対応
        #重み2=中間ノード数(15)と出力ノード(3)に対応
        weight1_var = tf.Variable(tf.truncated_nomal(
            [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_nomal(
            [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        #TODOバイアスの構造を設定
        #バイアス1は中間層へ伝達されるバイアス
        #バイアス2は出力層へ伝達されるバイアス
        bias1_var = tf.Variable(tf.truncated_nomal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.truncated_nomal([num_of_output_nodes], stddev=0.1), name="bias2")

    #入力データの転置を行う
    #input_phは入力データをテンソルにしたもの
    in1 = tf.transpose(input_ph)
    #転置したテンソルをreshapeする（reshapeってなに？）
    in2 = tf.reshape(in1, [-1, num_of_input_nodes])
    #入力層のノード計算
    in3 = tf.matmul(in2, weight1_var) + bias1_var

    #なにをしてるのだろう...?
    in4 = tf.split(in3, length_ofsequences, 0)

    #LSTMのCellを設定
    cell = tf.rnn_cell.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=false)
    rnn_output, states_op = tf.contrib.rnn.static_rnn(cell, in4, initial_state=istate_ph)
    output_op = tf.matmul(rnn_output[-1], weight2weight2_var) + bias2_var

    #データ取得のための操作
    w1_hist = tf.summary.histogram("weights1", weight1_var)
    w2_hist = tf.summary.histogram("weights2", weight2_var)
    b1_hist = tf.summary.histogram("biases1", bias1_var)
    b2_hist = tf.summary.histogram("biases2", bias2_var)
    output_hist = tf.summary.histogram("output",  output_op)
    results = [weight1_var, weight2_var, bias1_var,  bias2_var]
    return output_op, states_op, results

#損失関数(とりあえず2乗和誤差を用いてみる)
def loss(output_op, supervisor_ph):
    with tf.name_scope("loss") as scope:
        square_error = tf.reduce_mean(tf.square(output_op))
        loss_op = square_error
        tf.summary.scalar("loss", loss_op)
        return loss_op

#パラメータ更新
def traning(loss_op):
    with tf.name_scope("traning") as scope:
        traning_op = oprimizer.minimize(loss_op)
        return traning_op

#正解率算出
def calc_accuracy(output_op, prints=False):
    inputs, ts = make_prediction(num_of_prediction_epochs)
    pred_dict = {
        input_ph: inputs,
        supervisor_ph: ts,
        istate_ph:  np.zeros((num_of_prediction_epochs, num_of_hidden_nodes * 2))
    }
    output = sess.run([output_op], feed_dict=pred_dict)

    def print_result(i, p, q):
        [print(list(x)[0]) for x in i]
        print("output: %f, correct: %d", %(p,d) )
    if prints:
        [print_result(i, p, q) for i, p, q in zip(inputs, output[0], ts)]

    opt = abs(output - ts)[0]
    total = sum([1 if x[0] < 0.05 else for x in opt])
    print("accuracy %f" % (total / float(len(ts))))
    return output

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

X, t = create_data(num_of_sample, length_ofsequences)
