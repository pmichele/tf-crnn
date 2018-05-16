#!/usr/bin/env python
__author__ = 'solivr'
import argparse
import os
import csv
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
import glob

from src.config import Params, Alphabet, import_params_from_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ft', '--csv_files_train', type=str, help='CSV filename for training',
                        nargs='*', default=None)
    parser.add_argument('-fe', '--csv_files_eval', type=str, help='CSV filename for evaluation',
                        nargs='*', default=None)
    parser.add_argument('-o', '--output_model_dir', type=str,
                        help='Directory for output', default='./estimator')
    parser.add_argument('-n', '--nb-epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('-g', '--gpu', type=str, help="GPU 0,1 or '' ", default='')
    parser.add_argument('-p', '--params-file', type=str, help='Parameters filename', default=None)
    args = vars(parser.parse_args())

    if args.get('params_file'):
        dict_params = import_params_from_json(json_filename=args.get('params_file'))
        parameters = Params(**dict_params)
    else:
        parameters = Params(train_batch_size=128, # 128
                            eval_batch_size=128,  # 128
                            learning_rate=1e-3,  # 1e-3 recommended
                            learning_decay_rate=0.95,
                            learning_decay_steps=5000,
                            epoch_size=None,
                            save_interval=5e3,
                            input_shape=(32, 200),
                            optimizer='adam',
                            keep_prob=0.8,
                            digits_only=False,
                            alphabet=Alphabet.LETTERS_DIGITS_EXTENDED,
                            alphabet_decoding='same',
                            csv_delimiter='\t',
                            csv_files_eval=args.get('csv_files_eval'),
                            csv_files_train=args.get('csv_files_train'),
                            output_model_dir=args.get('output_model_dir'),
                            n_epochs=args.get('nb_epochs'),
                            gpu=args.get('gpu')
                            )

    # assert type(parameters.csv_files_train) == list and type(parameters.csv_files_eval) == list and\
    #        len(parameters.csv_files_train) > 0 and len(parameters.csv_files_eval) > 0,\
    #         'Input CSV should be a list of files'

    #check input had conforming alphabet
    # params_alphabet = set(parameters.alphabet)
    # input_alphabet = set()
    # for filename in parameters.csv_files_train + parameters.csv_files_eval:
    #     with open(filename, encoding='latin1') as file:
    #         for line in file:
    #             input_alphabet.update(line.split(parameters.csv_delimiter)[1])
    #     for sep in '\n\r':
    #         input_alphabet.discard(sep)
    #     extra_chars = input_alphabet - params_alphabet
    #     assert len(extra_chars) == 0, '%s in %s' % (extra_chars, filename)


    model_params = {
        'Params': parameters,
    }

    parameters.export_experiment_params() #The parameters are saved in the output_dir. Useful if the args were written in the terminal 

    os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
    config_sess = tf.ConfigProto()
    #config_sess.gpu_options.per_process_gpu_memory_fraction = 1.0
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

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,     #Create the pipeline 
                                       params=model_params,
                                       model_dir=parameters.output_model_dir,
                                       config=est_config
                                       )

    #SAMPLES_PER_FILE = 10000 # that's how we generated them for with-corpus3

    #Count number of image filenames in csv

    n_samples_eval = 0

    record_iterator = tf.python_io.tf_record_iterator(path=glob.glob(parameters.tfrecords_eval)[0]) 

    for i,string_record in enumerate(record_iterator):
        n_samples_eval += 1 

    print("n_samples_eval", n_samples_eval*len(glob.glob(parameters.tfrecords_eval)))

    # for file in parameters.csv_files_eval:
    #     with open(file, 'r', encoding='latin1') as csvfile:
    #         reader = csv.reader(csvfile, delimiter=parameters.csv_delimiter)
    #         n_samples_eval += len(list(reader))

    # if parameters.epoch_size is None:
    #     parameters.epoch_size = len(parameters.csv_files_train) * SAMPLES_PER_FILE
    # else:
    #     assert parameters.epoch_size <= len(parameters.csv_files_train) * SAMPLES_PER_FILE,\
    #            'Epoch size too big'


    
    #files_per_epoch = parameters.epoch_size // SAMPLES_PER_FILE #floor division
    try:
        for e in trange(0, parameters.n_epochs):
            # now we always evaluate every (sub-)epoch
            #epoch_train_subset = slice(e * files_per_epoch, (e+1) * files_per_epoch)
            #print(epoch_train_subset)
            estimator.train(input_fn=data_loader(tfrecords_filename=glob.glob(parameters.tfrecords_train),
                                                 params=parameters,
                                                 batch_size=parameters.train_batch_size,
                                                 num_epochs=1,
                                                 data_augmentation=True,
                                                 image_summaries=True))
            print('Train done')
            estimator.evaluate(input_fn=data_loader(tfrecords_filename=glob.glob(parameters.tfrecords_eval),
                                                    params=parameters,
                                                    batch_size=parameters.eval_batch_size,
                                                    num_epochs=1),
                               steps=np.floor(n_samples_eval/parameters.eval_batch_size)
                               )
            print('Eval done')
            
        
            estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                preprocess_image_for_prediction(fixed_height=parameters.input_shape[0], min_width=10))


    except KeyboardInterrupt:
        print('Interrupted')
        # estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
        #                             preprocess_image_for_prediction(fixed_height=parameters.input_shape[0], min_width=10))
        # print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))

    estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                preprocess_image_for_prediction(fixed_height=parameters.input_shape[0], min_width=10))
    print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))
