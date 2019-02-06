import tensorflow as tf
import loader
import numpy as np
import cv2

from loader import PredictionModel

from glob import glob
import os


"""m.compute_metrics("/notebooks/Detection/workspace/models/E2E/cer_48/zoom_in/tf1.12/predictions/", "/notebooks/Transcription/models/best_model/", '/notebooks/data/eval/val2_z
   ...: oom_in/val2_aligned_zoom_in.tfrecord', '/notebooks/Transcription/tf-crnn/model_params.json')"""

def compute_metrics(data_dir, model_dir, eval_tfrecord, parameters):
	pred = PredictionModel(model_dir, tf.Session(), parameters=parameters)
	tfrecords = glob(data_dir + "*.tfrecord")
	acc = 0.0
	i, fields_across_images, retrieved_fields_across_images = 0, 0, 0
	for tfrecord in tfrecords:
		CER, fields_retrieved, total_fields = compute_tfrecord_metrics(tfrecord, pred)
		fname = os.path.splitext(os.path.split(tfrecord)[1])[0]
		fname = fname[3:(len(fname) - 2)]
		i += 1
		print("done", i, fname, CER, fields_retrieved)
		acc += CER
		fields_across_images += total_fields
		retrieved_fields_across_images += fields_retrieved
	CER = acc / fields_across_images
	recall = _compute_recall(retrieved_fields_across_images, eval_tfrecord)
	print("total", CER, "recall", recall)
	return CER, recall

def _compute_recall(retrieved_fields, eval_tfrecord):
	acc = 0.0
	i = 0
	for example in tf.python_io.tf_record_iterator(eval_tfrecord):
		result = tf.train.Example.FromString(example)
		source_id = result.features.feature['image/source_id'].bytes_list.value[0].decode('latin1')
		labels = result.features.feature['image/object/transcription'].bytes_list.value
		acc += len(labels)
		i = i + 1
	return retrieved_fields / acc



def compute_tfrecord_metrics(tfrecord, predictor):
	acc, positives_count = 0.0, 0
	i = 0
	for example in tf.python_io.tf_record_iterator(tfrecord):
		result = tf.train.Example.FromString(example)
		label = result.features.feature['label'].bytes_list.value[0]
		image = np.fromstring(result.features.feature['image_raw'].bytes_list.value[0], np.uint8)
		image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
		image = np.expand_dims(image, axis=2)
		corpus = result.features.feature['corpus'].int64_list.value
		CER, positive = predictor.compute_CER_and_recall(image, corpus, label)
		# print("example:", CER)
		acc += CER
		positives_count += positive
		i = i + 1
	#return acc / i, positives_count
	return acc, positives_count, i
