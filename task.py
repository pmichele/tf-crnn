import argparse
import json
import os
import sys; print(sys.version)
try:
    import better_exceptions
except ImportError:
    pass
import tensorflow as tf
from .src.model import crnn_fn
from .src.data_handler import make_input_fn
from .src.data_handler import preprocess_image_for_prediction

from .src.config import Params, import_params_from_json

JSON_PARAMS = """ {
  "save_interval": 5000.0,
  "learning_rate": 0.001,
  "learning_rate_decay" : 0.95,
  "learning_rate_steps" : 5000,
  "optimizer": "adam",
  "keep_prob": 0.8,
  "eval_batch_size": 512,
  "train_batch_size": 512,
  "input_shape": [ 32, 256 ],
  "num_corpora": 10,
  "output_model_dir": "/notebooks/hwr/local-models/",
  "tfrecords_train": "/notebooks/hwr/tfrecords_data/2M_noise/*",
  "tfrecords_eval": "/notebooks/hwr/tfrecords_data/100k_constatlike/train/*",
  "alphabet": "letters_digits_extended",
  "alphabet_decoding": "same",
  "train_cnn": 1
}"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('-o', '--output_model_dir', type=str, required=False, help='Directory for output')
    parser.add_argument('-lr', '--learning-rate',   type=float, required=False, help='What is says on the tin')
    parser.add_argument('-lrd', '--learning-rate-decay', type=float, required=False, help='What is says on the tin')
    parser.add_argument('-lrs', '--learning-rate-steps', type=float, required=False, help='What is says on the tin')

    parser.add_argument('--job-dir',                type=str, required=False, help='UNUSED')
    parser.add_argument('--verbosity', default='INFO', type=str, required=False, help='TF verbosity')

    args = vars(parser.parse_args())
    args = dict(filter(lambda kv: kv[1] is not None, args.items()))
    tf.logging.set_verbosity(args['verbosity'])

    dict_params = json.loads(JSON_PARAMS)
    dict_params.update(args)
    parameters = Params(**dict_params)

    model_params = {
        'Params': parameters,
    }


    # Config estimator
    est_config = tf.estimator.RunConfig().replace(
        keep_checkpoint_max=200,
        save_checkpoints_steps=parameters.save_interval,
        save_checkpoints_secs=None,
        save_summary_steps=1000,
        model_dir=parameters.output_model_dir
    )

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,     # Create the pipeline
                                       params=model_params,
                                       model_dir=parameters.output_model_dir,
                                       config=est_config
                                       )

    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(parameters.tfrecords_train,
                               parameters.train_batch_size,
                               parameters.input_shape),
        max_steps=10 * (2e6 / parameters.train_batch_size))

    final_exporter = tf.estimator.FinalExporter('final',
        preprocess_image_for_prediction(fixed_height=parameters.input_shape[0],
                                        min_width=10))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(parameters.tfrecords_eval,
                               parameters.eval_batch_size,
                               parameters.input_shape,
                               repeat=False),
        exporters=[final_exporter])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
