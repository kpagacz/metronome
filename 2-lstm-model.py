import tensorflow as tf 
import csv 
import numpy as np 
import pandas as pd 
import functools

from config import RECORD_LENGTH, BATCH_SIZE

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


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


###########
# Baseline
###########
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numeric_columns),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

model.save_weights("./checkpoints/baseline-model-weights")



#############
# LSTM
#############
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numeric_columns),
    tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1)),
    tf.keras.layers.LSTM(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

train_data = packed_train_data.shuffle(250)
test_data = packed_test_data

model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

model.save_weights("./checkpoints/lstm-baseline")


###########
# CNN
###########
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numeric_columns),
    tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

train_data = packed_train_data.shuffle(250)
test_data = packed_test_data

model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

model.save_weights("./checkpoints/cnn-baseline")


###########
# CNN - complicated
###########
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numeric_columns),
    tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
    tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation="relu"),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

train_data = packed_train_data.shuffle(250)
test_data = packed_test_data

model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted being real glucose: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("REAL" if bool(survived) else "ADDITIONAL"))

model.save_weights("./checkpoints/cnn-v2")

###########
# CNN - inceptions
###########
# TODO (konrad.pagacz@gmail.com) go back to the DenseFeatures and figure out a way to use it
feature_layer_inputs = []
for header in NUMERIC_FEATURES:
  feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)

def InceptionLayer(layer, filters):
    # 1 kernel
    path1 = tf.keras.layers.Conv1D(filters[0], kernel_size=1, activation="relu")(layer)

    # 2 kernel
    path2 = tf.keras.layers.Conv1D(filters[1][0], kernel_size=1, activation="relu")(layer)
    path2 = tf.keras.layers.Conv1D(filters[1][1], kernel_size=2, activation="relu")(path2)

    # 3 kernel
    path3 = tf.keras.layers.Conv1D(filters[2][0], kernel_size=1, activation="relu")(layer)
    path3 = tf.keras.layers.Conv1D(filters[2][1], kernel_size=3, activation="relu")(path3)

    # 4 kernel
    path4 = tf.keras.layers.Conv1D(filters[3][0], kernel_size=1, activation="relu")(layer)
    path4 = tf.keras.layers.Conv1D(filters[3][1], kernel_size=4, activation="relu")(path4)

    # 5 kernel
    path5 = tf.keras.layers.Conv1D(filters[4][0], kernel_size=1, activation="relu")(layer)
    path5 = tf.keras.layers.Conv1D(filters[4][1], kernel_size=5, activation="relu")(path5)

    return tf.keras.layers.Concatenate(axis=1)([path1, path2, path3, path4])


def get_model():
    feature_layer = tf.keras.Input(shape=(BATCH_SIZE,))

    layer = tf.keras.layers.Dense(units=256)(feature_layer)
    layer = tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1))(layer)
    layer = InceptionLayer(layer, [32, (128, 32), (128, 32), (128, 32), (128, 32)])
    layer = InceptionLayer(layer, [32, (128, 32), (128, 32), (128, 32), (128, 32)])
    layer = tf.keras.layers.MaxPool1D(pool_size=3, strides=1)(layer)

    # Dense
    layer = tf.keras.layers.Dense(units=128)(layer)
    layer = tf.keras.layers.Dense(units=64)(layer)
    layer = tf.keras.layers.Dense(units=32)(layer)

    layer = tf.keras.layers.Dense(units=1)(layer)
    model = tf.keras.models.Model(inputs=feature_layer, outputs=layer)
    return model


model = get_model()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

train_data = packed_train_data.shuffle(250)
test_data = packed_test_data

model.fit(train_data, epochs=1)

test_loss, test_accuracy = model.evaluate(test_data)

print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted being real glucose: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("REAL" if bool(survived) else "ADDITIONAL"))

model.save_weights("./checkpoints/cnn-v2")