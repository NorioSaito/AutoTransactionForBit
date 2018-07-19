import tensorflow as tf
import numpy as np

num_of_input_nodes = 8 #入力層のノード数（日付を抜いた要素数）
num_of_hidden_nodes = 15 #中間層のノード数（とりあえず15でのちに調整する）
num_of_output_nodes = 3 #出力層のノード数(価格が「上がる」「下がる」「横這い」の3ユニット)
num_of_traning_epochs = 5000 #トレーニングの繰り返し回数(過学習が起きないよう調整する)
size_of_mini_batch = 150 #ミニバッチあたりのサンプル数(15x10で設定)
learning_rate = 0.01 #学習率(とりあえず0.01でやってみる)
forget_bias = 1 #よくわからない（デフォルトが1らしい）
num_of_sample = 7000 #データサンプル数

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
    #TODO
    #reshape調査（引数が何を設定するのか）
