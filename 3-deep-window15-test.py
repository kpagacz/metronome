import tensorflow as tf 
import csv 
import numpy as np 
import pandas as pd 
import functools
import errno
import os

import matplotlib.pyplot as plt


import sklearn

import itertools


from config import RECORD_LENGTH, BATCH_SIZE


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Setup GPU
print("Number of GPUs available: {}".format(len(tf.config.experimental.list_physical_devices("GPU"))))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical PGUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Dataset from csv
LABEL_COLUMN = "label"
LABELS = [0, 1]

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=32,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
  return dataset

TEST_FILE_PATH = "lstm-data-eval-15-variables.csv"
TEST_REVERSE_PATH = "lstm-data-eval-15-variables-reverse.csv"

raw_test_data = get_dataset(TEST_FILE_PATH, shuffle=False)
raw_reverse_test_data = get_dataset(TEST_REVERSE_PATH, shuffle=False)

# Packing numeric features
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names 

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feature, tf.float32) for feature in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features["numeric"] = numeric_features

        return features, labels


NUMERIC_FEATURES = ["var" + str(i) for i in range(RECORD_LENGTH - 1)]


packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_reverse_test_data = raw_reverse_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

# Finding mean and std for data normalization
MIN = 0
MAX = 350


def min_max_scale_numeric_data(data, min_, max_):
    return (data - min_) / max_ - min_

SCALER = functools.partial(min_max_scale_numeric_data, min_ = MIN, max_ = MAX)

numeric_column = tf.feature_column.numeric_column("numeric", normalizer_fn=SCALER, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

########################
# Model setup
########################
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=8554*50,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)


def compile(model, optimizer=None):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(
                          from_logits=True, name="binary_crossentropy"),
                      "accuracy"])



# Dense
model_dense = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numeric_columns),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])


#################
# TESTING
#################
test_data = packed_test_data
reverse_test_data = packed_reverse_test_data


compile(model_dense)
model_dense.load_weights("checkpoints/window15-deep")


evaluation = model_dense.evaluate(test_data)
probabilities = tf.sigmoid(model_dense.predict(test_data))

# Probabilities, true labels
def evaluate_model(model, test_data):
    true_labels = []
    predicted_proba = []
    for test_row, label in test_data:
        probas = (tf.sigmoid(model.predict(test_row)))
        probas = np.array(tf.squeeze(probas))

        labels = np.array(tf.squeeze(label))

        predicted_proba.append(list(probas))
        true_labels.append(list(labels))


    predicted_proba = np.fromiter(itertools.chain.from_iterable(predicted_proba), np.float, count=-1)
    true_labels = np.fromiter(itertools.chain.from_iterable(true_labels), np.float, count=-1)
    return predicted_proba, true_labels


proba, true = evaluate_model(model_dense, test_data)
predictions = (proba > 0.5).astype(np.int)

proba_reverse, true_reverse = evaluate_model(model_dense, reverse_test_data)
predictions_reverse = (proba_reverse > 0.5).astype(np.int)

proba_reverse_cut = proba_reverse[RECORD_LENGTH - 1:]
true_reverse_cut = true_reverse[RECORD_LENGTH - 1:]

proba_cut = proba[:-1 * (RECORD_LENGTH - 1)]
true_cut = true[:-1 * (RECORD_LENGTH - 1)]

voted_proba = voting(proba_cut, proba_reverse_cut, weights=[0.5, 0.5])
voted_predictions = (voted_proba > 0.5).astype(np.int)

def voting(model1_proba, model2_proba, weights: list):
    return np.average([model1_proba, model2_proba], weights=weights, axis=0)


# ROC
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(true, proba)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange",
    lw=lw, label="ROC curve (area = {:.2f}".format(roc_auc))
plt.plot([0,1], [0,1], color="navy", lw=lw, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("deep-window15-roc.png",
    bbox_inches="tight")
plt.show()
plt.close()

# Confusion
report = metrics.classification_report(true, predictions)
print(report)


##############
# Voted roc
##############
# ROC
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(true_cut, voted_proba)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange",
    lw=lw, label="ROC curve (area = {:.2f}".format(roc_auc))
plt.plot([0,1], [0,1], color="navy", lw=lw, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("deep-window15-voted-forward-reverse-roc.png",
    bbox_inches="tight")
plt.show()
plt.close()

# Confusion
report = metrics.classification_report(true_cut, voted_predictions)
print(report)
accuracy = metrics.accuracy_score(true_cut, voted_predictions)
print("Accuracy: {:.4f}".format(accuracy))