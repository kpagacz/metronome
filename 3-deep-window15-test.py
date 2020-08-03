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

TRAIN_FILE_PATH = "lstm-data-train-15-variables.csv"
TEST_FILE_PATH = "lstm-data-eval-15-variables.csv"

raw_train_data = get_dataset(TRAIN_FILE_PATH) 
raw_test_data = get_dataset(TEST_FILE_PATH)

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))
    print("{:20s}: {}".format("Label", label.numpy()))

print("RAW DATASET BATCH")
show_batch(raw_train_data)

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

packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

example_batch, labels_batch = next(iter(packed_train_data))

show_batch(packed_train_data)

# Finding mean and std for data normalization
desc = pd.read_csv(TRAIN_FILE_PATH)[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T["mean"])
STD = np.array(desc.T["std"])

MIN = 0
MAX = 350

def normalize_numeric_data(data, mean, std):
    return (data - mean) / std

def min_max_scale_numeric_data(data, min_, max_):
    return (data - min_) / max_ - min_


NORMALIZER = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
SCALER = functools.partial(min_max_scale_numeric_data, min_ = MIN, max_ = MAX)

numeric_column = tf.feature_column.numeric_column("numeric", normalizer_fn=SCALER, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column

example_batch["numeric"]

numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()


########################
# Callbacks and history
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


################
# TESTING
################
# Dense
model_densev1 = tf.keras.Sequential([
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

model_densev2 = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numeric_columns),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

model_dense = model_densev2
test_data = packed_test_data
compile(model_dense)
model_dense.load_weights("saved_models/DenseComplicatedv2")


evaluation = model_dense.evaluate(test_data)
probabilities = tf.sigmoid(model_dense.predict(test_data))

def evaluate_model(model, test_data):
    true_labels = []
    predicted_proba = []
    for test_row, label in test_data:
        probas = (tf.sigmoid(model.predict(test_row)))
        probas = np.array(tf.squeeze(probas))

        labels = np.array(tf.squeeze(label))

        predicted_proba.append(list(probas))
        true_labels.append(list(labels))


    predicted_proba = list(itertools.chain.from_iterable(predicted_proba))
    true_labels = list(itertools.chain.from_iterable(true_labels))
    return predicted_proba, true_labels

proba, true = evaluate_model(model_dense, test_data)

proba = np.array(proba)
true = np.array(true)
predictions = (proba > 0.5).astype(np.int)

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
plt.savefig("Densev2-roc.png",
    bbox_inches="tight")
plt.show()
plt.close()

# Confusion
report = metrics.classification_report(true, predictions)
print(report)
