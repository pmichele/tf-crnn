#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf
from config import Params, import_params_from_json


class PredictionModel:

    def __init__(self, model_dir, session, parameters=None):
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        self.model = tf.saved_model.loader.load(self.session, ['serve'], model_dir)

        self._input_dict, self._output_dict = _signature_def_to_tensors(self.model.signature_def['predictions'])

        if parameters:
            dict_params = import_params_from_json(json_filename=parameters)
            parameters = Params(**dict_params)
            keys = [c for c in parameters.alphabet.encode('latin1')]
            values = parameters.alphabet_codes
            init = tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64,
                        value_dtype=tf.int64)
            self.table_str2int = tf.contrib.lookup.HashTable(
                    init, -1)
            session.run(self.table_str2int.init)

    def predict(self, image, corpus):
        return self.session.run(self._output_dict,
                                feed_dict={ self._input_dict['images']: image,
                                            self._input_dict['corpora']: corpus,
                                })
    def compute_CER_and_recall(self, image, corpus, label):
        pred_tens = self._output_dict['words'][0]
        prediction = self.str2code(pred_tens)
        label_code = self.str2code(tf.constant([label]))
        CER = tf.edit_distance(tf.cast(prediction, dtype=tf.int64),
                 tf.cast(label_code, dtype=tf.int64))
        CER, words = self.session.run([CER, pred_tens], feed_dict={
            self._input_dict['images']: image,
            self._input_dict['corpora']: corpus,
            })
        print(words[0], label, CER)
        if words[0] == label:
            positive = 1
        else:
            positive = 0
        # with open('/notebooks/debug_metrics.txt', 'w+') as f:
        #     f.write(dic['words'][0][0])
        #     f.write(label)
        #     f.write(CER)
        return CER, positive


    def str2code(self, labels):
        with tf.name_scope('str2code_conversion'):
            table_str2int = self.table_str2int
            splitted = tf.string_split(labels, delimiter='')
            # values_int = tf.cast(tf.squeeze(tf.decode_raw(splitted.values, tf.uint8)), tf.int64) # Why the squeeze? it causes a bug
            values_int = tf.reshape(tf.cast(tf.decode_raw(splitted.values, tf.uint8), tf.int64), [-1])
            # values_int = tf.Print(values_int, [tf.shape(splitted.values)], message="splitted.values", summarize=9999)
            codes = table_str2int.lookup(values_int)
            codes = tf.cast(codes, tf.int32)
            return tf.SparseTensor(splitted.indices, codes, splitted.dense_shape)

def _signature_def_to_tensors(signature_def):  # from SeguinBe
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}
