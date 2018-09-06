import argparse

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp
import source.developer as developer
developer.print_stamp()

def main():
    training_keys = ['100']
    dataset = dman.DataSet(key_tr=training_keys)
    lstm = nn.LSTM_Model(batch_size=FLAGS.batch, data_dim=dataset.data_dim)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    tfp.training(sess=sess, neuralnet=lstm, saver=saver, dataset=dataset, batch_size=FLAGS.batch, sequence_length=FLAGS.trainlen, iteration=FLAGS.iter)
    tfp.validation(sess=sess, neuralnet=lstm, saver=saver, dataset=dataset, sequence_length=FLAGS.testlen)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=100, help='-') # 100 batch * 36 sequence = 10 sec
    parser.add_argument('--trainlen', type=int, default=100, help='-')
    parser.add_argument('--testlen', type=int, default=50, help='-')
    parser.add_argument('--iter', type=int, default=3000, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
