#!/usr/bin/env python
__author__ = 'solivr'
import argparse
import os
import numpy as np
try:
    import better_exceptions
except ImportError:
    pass
import tensorflow as tf
from .model import crnn_fn
from .data_handler import make_input_fn
from .data_handler import preprocess_image_for_prediction

from .config import Params, import_params_from_json

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
        save_summary_steps=2000,
        model_dir=parameters.output_model_dir
    )

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,     # Create the pipeline
                                       params=model_params,
                                       model_dir=parameters.output_model_dir,
                                       config=est_config
                                       )

    try:
        for e in range(0, parameters.n_epochs):
            estimator.train(input_fn=make_input_fn(parameters.tfrecords_train,
                                                   parameters.train_batch_size,
                                                   parameters.input_shape,
                                                   dynamic_distortion=parameters.dynamic_distortion,
                                                   repeat=False),

                            )
            print('Train done')
            estimator.evaluate(input_fn=make_input_fn(parameters.tfrecords_eval,
                                                      parameters.eval_batch_size,
                                                      parameters.input_shape,
                                                      dynamic_distortion=False,
                                                      repeat=False)
                               )
            print('Eval done')


            estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                preprocess_image_for_prediction(fixed_height=parameters.input_shape[0], min_width=10))

    except KeyboardInterrupt:
        print('Interrupted')

    estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                preprocess_image_for_prediction(fixed_height=parameters.input_shape[0], min_width=10))
    print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))
