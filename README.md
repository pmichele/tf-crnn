# Convolutional Recurrent Neural Network in Tensorflow (tf.crnn)
CRNN model in Tensorflow using Estimators

Implementation of the Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR.
Original paper http://arxiv.org/abs/1507.05717 and code https://github.com/bgshih/crnn

This version uses the `tf.estimator.Estimator` to build the model.

### Contents
* `src/model.py` : definition of the model
* `src/data_handler.py` : functions for data loading, preprocessing and data augmentation
* `src/config.py` : `class Params` manages parameters of model and experiments
* `src/decoding.py` : helper fucntion to transform characters to words
* `train.py` : script to launch for training the model, more info on the parameters and options inside
* `export_model.py`: script to export a model once trained, i.e for serving
* Extra : `hlp/numbers_mnist_generator.py` : generates a sequence of digits to form a number using the MNIST database
* Extra : `hlp/csv_path_convertor.py` : converts a csv file with relative paths to a csv file with absolute paths

### How to train a model on Google ML-Engine

To try locally, just run `python3 -m tf_crnn.task` in the directory above `tf_crnn`

To run on the cloud, you need to submit a job. For this, the code needs to be packaged, which is achieved via the `setup.py` file.
However, this should reside in the directory **above** `tf_crnn`, so issue the command:
```
    mv tf_crnn/_setup.py tf_crnn/../setup.py
```

Then, in that parent directory, you need to run the `glcoud` command. For simplicity, we hard-coded the parameters in the `task.py`
file. **Some** of them can be overriden via command line parameters. If we need hyper-parameter search, more command line args
should be added, i.e. learning rate.

```
    gcloud ml-engine jobs submit training hwr-$(date +%Y%m%d_%H%M%S) \
    --job-dir gs://hwr-data//models/hwr_p100/ \
    --staging-bucket gs://hwr-data \
    --module-name tf_crnn.task \
    --region europe-west1 \
    --runtime-version 1.8 \
    --config config_p100.yaml \
    -- \
    --verbosity DEBUG
```

### Dependencies
* `tensorflow` (1.3)
* `tensorflow-tensorboard` (0.1.7) (not mandatory but useful to visualise loss, accuracy and inputs / outputs)
* `tqdm` for progress bars
* `json`
* `glob`



