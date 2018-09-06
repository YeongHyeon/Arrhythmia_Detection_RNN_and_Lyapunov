import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

class LSTM_Model(object):

    def __init__(self, batch_size=1, data_dim=None):

        print("\n** Initialize the LSTM Model")

        self.inputs = tf.placeholder(tf.float32, [None, None, data_dim])
        self.outputs = tf.placeholder(tf.float32, [None, None, data_dim])

        batch_size = int(batch_size)

        with tf.variable_scope('lstm'):
            # Activated by tanh
            self.cell_1 = tf.contrib.rnn.LSTMCell(num_units=data_dim, initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0)
            self.cell_2 = tf.contrib.rnn.LSTMCell(num_units=data_dim, initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0)
            self.cell_3 = tf.contrib.rnn.LSTMCell(num_units=data_dim, initializer=tf.contrib.layers.xavier_initializer(), forget_bias=1.0)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell_1, self.cell_2, self.cell_3])

            # self.initial_state = self.cell.zero_state(batch_size=batch_size, dtype=tf.float32) # It requires same batchsize at training and vaidation
            self.logits, states = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.inputs, dtype=tf.float32)

        self.vars_lstm = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')
        self.loss = tf.sqrt(tf.reduce_sum(tf.square(self.logits - self.outputs)))
        self.train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=self.loss, var_list=self.vars_lstm)

        tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()
