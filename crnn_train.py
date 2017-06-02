#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf
import numpy as np
import time
from model import CRNN, CTC
from config import Conf
from dataset import Dataset
from decoding import simpleDecoder, evaluation_metrics


# CONFIG PARAMETERS
# -----------------

# Limit the usage of GPU memory to 30%
config_sess = tf.ConfigProto()
config_sess.gpu_options.per_process_gpu_memory_fraction = 0.3

config = Conf(n_classes=37,
              train_batch_size=128,
              test_batch_size=32,
              learning_rate=0.001,  # 0.001 for adadelta
              decay_rate=0.9,
              max_iteration=3000000,
              max_epochs=100,
              eval_interval=200,
              save_interval=2500,
              file_writer='../rms_d09',
              data_set='/home/soliveir/NAS-DHProcessing/mnt/ramdisk/max/90kDICT32px/',
              model_dir='../model-crnn-rms_d09/',
              input_shape=[32, 100],
              list_n_hidden=[256, 256],
              max_len=24)

session = tf.Session(config=config_sess)


def crnn_train(conf=config, sess=session):

    # PLACEHOLDERS
    # ------------

    # Sequence length, parameter of stack_bidirectional_dynamic_rnn,
    rnn_seq_len = tf.placeholder(tf.int32, [None], name='sequence_length')
    target_seq_len = tf.placeholder(tf.int32, [None], name='target_seq_len')
    input_ctc_seq_len = tf.placeholder(tf.int32, [None], name='input_ctc_seq_len')
    is_training = tf.placeholder(tf.bool, name='trainable')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, conf.inputShape[0], conf.inputShape[1], 1], name='input')
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    labels = tf.placeholder(tf.int32, [None], name='labels')

    # Sequence length
    train_seq_len = [conf.maxLength for _ in range(conf.trainBatchSize)]
    test_seq_len = [conf.maxLength for _ in range(conf.testBatchSize)]

    # # Evaluation
    # true_string = tf.placeholder(tf.string, [conf.testBatchSize, None], name='true_label_string')
    # predicted_string = tf.placeholder(tf.string, [conf.testBatchSize, None], name='predicted_string')

    # NETWORK
    # -------

    # Network and ctc definition
    crnn = CRNN(x, conf, rnn_seq_len, is_training, keep_prob, session=sess)
    ctc = CTC(crnn.prob, labels, target_seq_len, inputSeqLength=input_ctc_seq_len)

    # Optimizer defintion
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(conf.learning_rate, global_step, 5000,
                                               conf.decay_rate, staircase=True)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(ctc.loss, global_step=global_step)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(ctc.loss, global_step=global_step)

    # ERRORS EVALUATION
    # -----------------
    # accuracy, WER, CER = evaluation_metrics(predicted_string, true_string)


    # SUMMARIES
    # ---------

    # Cost
    tf.summary.scalar('cost', ctc.cost)
    # Learning rate
    tf.summary.scalar('learning_rate', learning_rate)
    # Accuracy
    accuracy = tf.placeholder(tf.float32, None, name='accuracy_var')
    tf.summary.scalar('accuracy', accuracy)
    # WER
    WER = tf.placeholder(tf.float32, None, name='WER')
    tf.summary.scalar('WER', WER)
    # CER
    CER = tf.placeholder(tf.float32, None, name='CER')
    tf.summary.scalar('CER', CER)

    # Summary Writer
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(conf.fileWriter,
                                         graph=tf.get_default_graph(),
                                         flush_secs=10)

    # tf.summary.FileWriter("./graph", graph=tf.get_default_graph(), flush_secs=10)

    # GRAPH
    # -----

    # Initialize graph
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0

    # Data
    data_train = Dataset(conf,
                         path=conf.dataSet,
                         mode='train')
    data_test = Dataset(conf,
                        path=conf.dataSet,
                        mode='val')

    # RUN SESSION
    # -----------

    start_time = time.time()
    t = start_time
    while step < conf.maxIteration:
        # Prepare batch and add channel dimension
        images_batch, label_set, seq_len = data_train.nextBatch(conf.trainBatchSize)
        images_batch = np.expand_dims(images_batch, axis=-1)

        cost, _, step = sess.run([ctc.cost, optimizer, global_step],
                                 feed_dict={
                                           x: images_batch,
                                           keep_prob: 0.7,
                                           rnn_seq_len: train_seq_len,
                                           input_ctc_seq_len: train_seq_len,
                                           target_seq_len: seq_len,
                                           labels: label_set[1],
                                           is_training: True,
                                        })

        # Eval accuarcy
        if step != 0 and step % conf.evalInterval == 0:
            images_batch_eval, label_set_eval, seq_len_eval = data_test.nextBatch(conf.testBatchSize)
            images_batch_eval = np.expand_dims(images_batch_eval, axis=-1)

            raw_pred = sess.run(crnn.rawPred,
                                feed_dict={
                                            x: images_batch_eval,
                                            keep_prob: 1.0,
                                            is_training: False,
                                            rnn_seq_len: test_seq_len,
                                            input_ctc_seq_len: test_seq_len,
                                            target_seq_len: seq_len_eval,
                                            labels: label_set_eval[1],
                                           })

            str_pred = simpleDecoder(raw_pred)
            # acc = eval_accuracy(str_pred, label_set_eval[0])
            # wer = eval_WER(str_pred, label_set_eval[0])
            # cer = eval_CER(str_pred, label_set_eval[0])
            acc, wer, cer = evaluation_metrics(str_pred, label_set_eval[0])

            print('step: {}, cost: {}, training accuracy: {}, cer: {}'.format(step, cost, acc, cer))

            # for i in range(5):
            #     print('original: {}, predicted(no decode): {}, predicted: {}'.format(label_set_eval[0][i],
            #                                                                          str_pred[i]))

            time_elapse = time.time() - t
            t = time.time()

            summary, step = sess.run([merged, global_step],
                                     feed_dict={
                                         x: images_batch,
                                         keep_prob: 0.7,
                                         rnn_seq_len: train_seq_len,
                                         input_ctc_seq_len: train_seq_len,
                                         target_seq_len: seq_len,
                                         labels: label_set[1],
                                         is_training: False,
                                         accuracy: acc,
                                         WER: wer,
                                         CER: cer
                                     })

            train_writer.add_summary(summary, step)

        if step != 0 and step % conf.saveInterval == 0:
            crnn.saveModel(conf.modelDir, step)

        if step >= conf.maxIteration:
            print('{} training has completed'.format(conf.maxIteration))
            crnn.saveModel(conf.modelDir, step)


if __name__ == '__main__':
    crnn_train()