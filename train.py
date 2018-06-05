#!/usr/bin/env python
__author__ = 'solivr'
import argparse
import os
import numpy as np
try:
    import better_exceptions
except ImportError:
    pass
from tqdm import trange
import tensorflow as tf
from src.model import crnn_fn
from src.data_handler import data_loader
from src.data_handler import preprocess_image_for_prediction
from glob import glob

from src.config import Params, Alphabet, import_params_from_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model according to the specified config in the JSON. '
                                     'The optional arguments override the ones from the file.')
    parser.add_argument('params_file',              type=str, help='Parameters filename (JSON)')
    parser.add_argument('-o', '--output_model_dir', type=str, required=False, help='Directory for output')
    parser.add_argument('-n', '--nb_epochs',        type=int, required=False, help='Number of epochs')
    parser.add_argument('-g', '--gpu',              type=str, required=False, help="GPU 0,1 or '' ")
    args = vars(parser.parse_args())
    args = dict(filter(lambda kv: kv[1] is not None, args.items()))

    dict_params = import_params_from_json(json_filename=args.get('params_file'))
    dict_params.update(args)
    parameters = Params(**dict_params)

    model_params = {
        'Params': parameters,
    }

    # The parameters are saved in the output_dir to keep their most recent copy,
    # including the overriding command line arguments
    parameters.export_experiment_params()

    os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True


    # Config estimator
    est_config = tf.estimator.RunConfig().replace(
        keep_checkpoint_max=200,
        save_checkpoints_steps=parameters.save_interval,
        session_config=config_sess,
        save_checkpoints_secs=None,
        save_summary_steps=1000,
        model_dir=parameters.output_model_dir
    )

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,     # Create the pipeline
                                       params=model_params,
                                       model_dir=parameters.output_model_dir,
                                       config=est_config
                                       )

    record_iterator = tf.python_io.tf_record_iterator(path=parameters.tfrecords_eval)
    for n_samples_eval,string_record in enumerate(record_iterator):
        pass
    n_samples_eval += 1

    print("n_samples_eval", n_samples_eval)

    try:
        for e in trange(0, parameters.n_epochs):
            estimator.train(input_fn=data_loader(tfrecords_filename=glob(parameters.tfrecords_train),
                                                 params=parameters,
                                                 batch_size=parameters.train_batch_size,
                                                 num_epochs=1,
                                                 data_augmentation=True,
                                                 image_summaries=True))
            print('Train done')
            estimator.evaluate(input_fn=data_loader(tfrecords_filename=glob(parameters.tfrecords_eval),
                                                    params=parameters,
                                                    batch_size=parameters.eval_batch_size,
                                                    num_epochs=1),
                               steps=max(n_samples_eval // parameters.eval_batch_size, 1) # minimum 1 in case n_samples eval < 512
                               )
            print('Eval done')


            estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                preprocess_image_for_prediction(fixed_height=parameters.input_shape[0], min_width=10))


    except KeyboardInterrupt:
        print('Interrupted')

    estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                preprocess_image_for_prediction(fixed_height=parameters.input_shape[0], min_width=10))
    print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))
