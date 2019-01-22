#!/usr/bin/env python
__author__ = 'solivr'


import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from .decoding import get_words_from_chars
from .config import  Params, CONST


def weightVar(shape, mean=0.0, stddev=0.02, name='weights'):
    init_w = lambda : tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(init_w, name=name)


def biasVar(shape, value=0.0, name='bias'):
    init_b = lambda : tf.constant(value=value, shape=shape)
    return tf.Variable(init_b, name=name)


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name=None):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding, name=name)


def deep_cnn(input_imgs: tf.Tensor, is_training: bool, summaries: bool=True, on_document=False) -> tf.Tensor:
    input_tensor = input_imgs
    if input_tensor.shape[-1] == 1:
        input_channels = 1
    elif input_tensor.shape[-1] == 3:
        input_channels = 3
    else:
        raise NotImplementedError

    # Following source code, not paper

    with tf.variable_scope('deep_cnn'):
        # - conv1 - maxPool2x2
        with tf.variable_scope('layer1'):
            W = weightVar([3, 3, input_channels, 64])
            b = biasVar([64])
            conv = conv2d(input_tensor, W, name='conv')
            out = tf.nn.bias_add(conv, b)
            conv1 = tf.nn.relu(out)
            pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool')

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer1/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer1/bias:0'][0]
                tf.summary.histogram('bias', bias)


        # - conv2 - maxPool 2x2
        with tf.variable_scope('layer2'):
            W = weightVar([3, 3, 64, 128])
            b = biasVar([128])
            conv = conv2d(pool1, W)
            out = tf.nn.bias_add(conv, b)
            conv2 = tf.nn.relu(out)
            pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer2/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer2/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv3 - w/batch-norm (as source code, not paper)
        with tf.variable_scope('layer3'):
            W = weightVar([3, 3, 128, 256])
            b = biasVar([256])
            conv = conv2d(pool2, W)
            out = tf.nn.bias_add(conv, b)
            b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                   training=is_training, name='batch-norm')
            conv3 = tf.nn.relu(b_norm, name='ReLU')

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer3/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer3/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv4 - maxPool 2x1
        with tf.variable_scope('layer4'):
            W = weightVar([3, 3, 256, 256])
            b = biasVar([256])
            conv = conv2d(conv3, W)
            out = tf.nn.bias_add(conv, b)
            conv4 = tf.nn.relu(out)
            pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                                   padding='SAME', name='pool4')

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer4/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer4/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv5 - w/batch-norm
        with tf.variable_scope('layer5'):
            W = weightVar([3, 3, 256, 512])
            b = biasVar([512])
            conv = conv2d(pool4, W)
            out = tf.nn.bias_add(conv, b)
            b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                   training=is_training, name='batch-norm')
            conv5 = tf.nn.relu(b_norm)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer5/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer5/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv6 - maxPool 2x1 (as source code, not paper)
        with tf.variable_scope('layer6'):
            W = weightVar([3, 3, 512, 512])
            b = biasVar([512])
            conv = conv2d(conv5, W)
            out = tf.nn.bias_add(conv, b)
            conv6 = tf.nn.relu(out)
            pool6 = tf.nn.max_pool(conv6, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                                   padding='SAME', name='pool6')

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer6/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer6/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv 7 - w/batch-norm (as source code, not paper)
        with tf.variable_scope('layer7'):
            W = weightVar([2, 2, 512, 512])
            b = biasVar([512])
            if on_document:
                conv = conv2d(pool6, W, padding='SAME', strides=[1, 2, 1, 1])
            else:
                # We assume the input has height 32, otherwise, well it will give unexpected results
                conv = conv2d(pool6, W, padding='VALID')
            out = tf.nn.bias_add(conv, b)
            b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                   training=is_training, name='batch-norm')
            conv7 = tf.nn.relu(b_norm)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/bias:0'][0]
                tf.summary.histogram('bias', bias)

        cnn_net = conv7

        if on_document:
            return cnn_net

        with tf.variable_scope('Reshaping_cnn'):
            shape = cnn_net.get_shape().as_list()  # [batch, height, width, features]
            transposed = tf.transpose(cnn_net, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [shape[0], -1, shape[1] * shape[3]],
                                       name='reshaped')  # [batch, width, height x features]

        return conv_reshaped


def deep_bidirectional_lstm(inputs: tf.Tensor, corpora: tf.Tensor, params: Params, summaries: bool=True) -> tf.Tensor:
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input) "(batch, time, height)"

    list_n_hidden = [256, 256]

    # add the corpora to all input times; TODO: what values should we use for one-hot? (0,1) ?

    with tf.name_scope('corpus_concat'):
        corpora = tf.expand_dims(corpora, axis=1) # add the time dimension
        corpora = tf.one_hot(corpora, depth=params.num_corpora, dtype=inputs.dtype, name='corpus_to_onehot')
        multiples = tf.stack([1, tf.shape(inputs)[1], 1])     #tf.shape(input)[1] = width

        corpora = tf.tile(corpora, multiples)
        inputs = tf.concat((corpora, inputs), axis=2, name='concat_corpus')

    with tf.name_scope('deep_bidirectional_lstm'):
        # Forward direction cells
        fw_cell_list = [LSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        # Backward direction cells
        bw_cell_list = [LSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

        lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                        bw_cell_list,
                                                                        inputs,
                                                                        dtype=tf.float32
                                                                        )

        # Dropout layer
        print('Using dropout', params.keep_prob_dropout)
        lstm_net = tf.nn.dropout(lstm_net, keep_prob=params.keep_prob_dropout)

        with tf.variable_scope('Reshaping_rnn'):
            shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
            rnn_reshaped = tf.reshape(lstm_net, [-1, shape[-1]])  # [batch x width, 2*n_hidden]

        with tf.variable_scope('fully_connected'):
            W = weightVar([list_n_hidden[-1]*2, params.n_classes])
            b = biasVar([params.n_classes])
            fc_out = tf.nn.bias_add(tf.matmul(rnn_reshaped, W), b)

            if summaries:
                weights = [var for var in tf.global_variables()
                           if var.name == 'deep_bidirectional_lstm/fully_connected/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables()
                        if var.name == 'deep_bidirectional_lstm/fully_connected/bias:0'][0]
                tf.summary.histogram('bias', bias)

        lstm_out = tf.reshape(fc_out, [-1, shape[1], params.n_classes], name='reshape_out')  # [batch, width, n_classes]

        raw_pred = tf.argmax(tf.nn.softmax(lstm_out), axis=2, name='raw_prediction')

        # Swap batch and time axis
        lstm_out = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

        return lstm_out, raw_pred


def crnn_fn(features, labels, mode, params):
    """
    :param features: dict {
                            'image'
                            'images_width'
                            'corpora'
                            }
    :param labels: labels. flattend (1D) array with encoded label (one code per character)
    :param mode:
    :param params: dict {
                            'Params'
                        }
    :return:
    """

    parameters = params.get('Params')
    assert isinstance(parameters, Params)

    if mode != tf.estimator.ModeKeys.TRAIN:
        parameters.keep_prob_dropout = 1.0

    conv = deep_cnn(features['image'], (mode == tf.estimator.ModeKeys.TRAIN), summaries=False)


    logprob, raw_pred = deep_bidirectional_lstm(conv, features['corpus'], params=parameters, summaries=False)

    # Compute seq_len from image width
    n_pools = CONST.DIMENSION_REDUCTION_W_POOLING  # 2x2 pooling in dimension W on layer 1 and 2
    seq_len_inputs = tf.divide(features['image_width'], n_pools, name='seq_len_input_op') - 1

    predictions_dict = {'prob': logprob,
                        'raw_predictions': raw_pred
                        }

    if not mode == tf.estimator.ModeKeys.PREDICT:
        # Alphabet and codes
        keys = [c for c in parameters.alphabet.encode('latin1')]
        values = parameters.alphabet_codes

        # Convert string label to code label
        with tf.name_scope('str2code_conversion'):
            table_str2int = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64, value_dtype=tf.int64), -1)
            splitted = tf.string_split(labels, delimiter='')
            values_int = tf.cast(tf.squeeze(tf.decode_raw(splitted.values, tf.uint8)), tf.int64)
            codes = table_str2int.lookup(values_int)
            codes = tf.cast(codes, tf.int32)
            sparse_code_target = tf.SparseTensor(splitted.indices, codes, splitted.dense_shape)

        seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32), #array of labels length
                                         minlength= tf.shape(predictions_dict['prob'])[1])

        # Loss
        # ----
        # >>> Cannot have longer labels than predictions -> error

        with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                      inputs=predictions_dict['prob'],
                                      sequence_length=tf.cast(seq_len_inputs, tf.int32),
                                      preprocess_collapse_repeated=False,
                                      ctc_merge_repeated=True,
                                      ignore_longer_outputs_than_inputs=True,  # returns zero gradient in case it happens -> ema loss = NaN
                                      time_major=True)
            loss_ctc = tf.reduce_mean(loss_ctc)
            loss_ctc = tf.Print(loss_ctc, [loss_ctc], message='* Loss : ')


        global_step = tf.train.get_or_create_global_step()
        # # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step, zero_debias=True)
        # Create the shadow variables, and add op to maintain moving averages
        maintain_averages_op = ema.apply([loss_ctc])
        loss_ema = ema.average(loss_ctc)


        # Train op
        # --------
        if parameters.learning_rate_decay:
            learning_rate = tf.train.exponential_decay(parameters.learning_rate, global_step,
                                                       parameters.learning_rate_steps,
                                                       parameters.learning_rate_decay, staircase=True)
        else:
            learning_rate = tf.constant(parameters.learning_rate)


        if parameters.optimizer == 'ada':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif parameters.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, epsilon=1e-07) # at 1e-08 sometimes exploding gradient
        elif parameters.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)

        if not parameters.train_cnn:
            trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'deep_bidirectional_lstm')
            print('Training LSTM only')
        else:
            trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        opt_op = optimizer.minimize(loss_ctc, global_step=global_step, var_list=trainable)

        with tf.control_dependencies(update_ops + [opt_op]):
            train_op = tf.group(maintain_averages_op)

        # Summaries
        # ---------
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('losses/ctc_loss', loss_ctc)
    else:
        loss_ctc, train_op = None, None

    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.TRAIN]:
        with tf.name_scope('code2str_conversion'):
            keys = tf.cast(parameters.alphabet_decoding_codes, tf.int64)
            values = [c for c in parameters.alphabet_decoding]
            table_int2str = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')

            sparse_code_pred, log_probability = tf.nn.ctc_beam_search_decoder(predictions_dict['prob'],
                                                                              sequence_length=tf.cast(seq_len_inputs, tf.int32),
                                                                              merge_repeated=False,
                                                                              beam_width=100,
                                                                              top_paths=parameters.nb_logprob)
            # confidence value

            predictions_dict['score'] = log_probability

            sequence_lengths_pred = [tf.bincount(tf.cast(sparse_code_pred[i].indices[:, 0], tf.int32),
                                                minlength=tf.shape(predictions_dict['prob'])[1]) for i in range(parameters.top_paths)]

            pred_chars = [table_int2str.lookup(sparse_code_pred[i]) for i in range(parameters.top_paths)]

            list_preds = [get_words_from_chars(pred_chars[i].values, sequence_lengths=sequence_lengths_pred[i])
                          for i in range(parameters.top_paths)]

            predictions_dict['words'] = tf.stack(list_preds)

            tf.summary.text('predicted_words', predictions_dict['words'][0][:10])

    # Evaluation ops
    # --------------
    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('evaluation'):
            CER = tf.metrics.mean(tf.edit_distance(sparse_code_pred[0], tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')

            # Convert label codes to decoding alphabet to compare predicted and groundtrouth words
            target_chars = table_int2str.lookup(tf.cast(sparse_code_target, tf.int64))
            target_words = get_words_from_chars(target_chars.values, seq_lengths_labels)
            accuracy = tf.metrics.accuracy(target_words, predictions_dict['words'][0], name='accuracy')

            eval_metric_ops = {
                               'eval/accuracy': accuracy,
                               'eval/CER': CER,
                               }
            CER = tf.Print(CER, [CER], message='-- CER : ')
            accuracy = tf.Print(accuracy, [accuracy], message='-- Accuracy : ')

    else:
        eval_metric_ops = None

    export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions_dict)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss_ctc,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        scaffold=tf.train.Scaffold()
        # scaffold=tf.train.Scaffold(init_fn=None)  # Specify init_fn to restore from previous model
    )
