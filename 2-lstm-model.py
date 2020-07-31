import tensorflow as tf 
import csv 
import numpy as np 
import pandas as pd 
import functools
import errno
import os


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
raw_test_data = get_dataset(TEST_FILE_PATH, shuffle=False)

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
  decay_steps=8554*30,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)


logdir = "logs"
try:
    os.makedirs(logdir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/DeepCleanerv2",
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_freq="epoch",
    verbose=1
)


def get_callbacks():
    return [
        # tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30),
        model_checkpoint_callback,
        # tf.keras.callbacks.TensorBoard(logdir + "/" + name)
    ]

def compile_and_fit(model, train_data, test_data, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[

                      "accuracy",
                      tf.keras.metrics.AUC()])


    history = model.fit(
        train_data,
        epochs=max_epochs,
        validation_data=test_data,
        callbacks=get_callbacks(),
        verbose=1
    )

    model.summary()

    return history


histories = {}

###############
# NAS
###############
# import autokeras as ak

# classifier = ak.StructuredDataClassifier(
#     loss="binary_crossentropy",
#     metrics=["AUC", "accuracy"],
#     objective="val_AUC",
#     overwrite=True,
#     max_trials=100,
#     tuner="greedy"
# )


# train_data = packed_train_data.shuffle(500)
# test_data = packed_test_data

# classifier.fit(
#     train_data,
#     validation_split=0.1,
#     epochs=1000
# )


###########
# Baseline
###########
# model = tf.keras.Sequential([
#     tf.keras.layers.DenseFeatures(numeric_columns),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     optimizer="adam",
#     metrics=["accuracy"]
# )

# train_data = packed_train_data.shuffle(500)
# test_data = packed_test_data

# model.fit(train_data, epochs=20)

# test_loss, test_accuracy = model.evaluate(test_data)

# print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

# model.save_weights("./checkpoints/baseline-model-weights")

#####################
# Dense complicated
#####################
# model_dense = tf.keras.Sequential([
#     tf.keras.layers.DenseFeatures(numeric_columns),
#     tf.keras.layers.Dense(2048, activation="relu"),
#     tf.keras.layers.Dense(1024, activation="relu"),
#     tf.keras.layers.Dense(1024, activation="relu"),
#     tf.keras.layers.Dense(1024, activation="relu"),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dense(256, activation="relu"),
#     tf.keras.layers.Dense(256, activation="relu"),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid"),
# ])

model_dense = tf.keras.Sequential([
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

model = model_dense
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=
    tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"])

model.fit(
    train_data,
    epochs=1000,
    validation_data=test_data,
    callbacks=get_callbacks()
)

test_loss, test_accuracy = model.evaluate(test_data)

print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

model.save_weights("./checkpoints/DenseComplicatedv2")

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted being real glucose: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("REAL" if bool(survived) else "ADDITIONAL"))

# #############
# # LSTM
# #############
# model = tf.keras.Sequential([
#     tf.keras.layers.DenseFeatures(numeric_columns),
#     tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1)),
#     tf.keras.layers.LSTM(32, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     optimizer="adam",
#     metrics=["accuracy"]
# )

# train_data = packed_train_data.shuffle(250)
# test_data = packed_test_data

# model.fit(train_data, epochs=20)

# test_loss, test_accuracy = model.evaluate(test_data)

# print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

# model.save_weights("./checkpoints/lstm-baseline")


# ###########
# # CNN
# ###########
# model = tf.keras.Sequential([
#     tf.keras.layers.DenseFeatures(numeric_columns),
#     tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1)),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     optimizer="adam",
#     metrics=["accuracy"]
# )

# train_data = packed_train_data.shuffle(250)
# test_data = packed_test_data

# model.fit(train_data, epochs=20)

# test_loss, test_accuracy = model.evaluate(test_data)

# print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

# model.save_weights("./checkpoints/cnn-baseline")


# ###########
# # CNN - complicated
# ###########
# model = tf.keras.Sequential([
#     tf.keras.layers.DenseFeatures(numeric_columns),
#     tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1)),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation="relu"),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu"),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=128),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     optimizer="adam",
#     metrics=["accuracy"]
# )

# train_data = packed_train_data.shuffle(250)
# test_data = packed_test_data

# model.fit(train_data, epochs=20)

# test_loss, test_accuracy = model.evaluate(test_data)

# print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

# predictions = model.predict(test_data)

# # Show some results
# for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
#   prediction = tf.sigmoid(prediction).numpy()
#   print("Predicted being real glucose: {:.2%}".format(prediction[0]),
#         " | Actual outcome: ",
#         ("REAL" if bool(survived) else "ADDITIONAL"))

# model.save_weights("./checkpoints/cnn-v2")

###########
# CNN - inceptions
###########
# feature_layer_inputs = {}
# for header in ["numeric"]:
#   feature_layer_inputs[header] = tf.keras.Input(shape=(14,), name=header)

# PADDING = "same"

# def InceptionLayer(layer, filters):
#     # 1 kernel
#     path1 = tf.keras.layers.Conv1D(filters[0], kernel_size=1, activation="relu")(layer)

#     # 2 kernel
#     path2 = tf.keras.layers.Conv1D(filters[1][0], kernel_size=1, activation="relu", padding=PADDING)(layer)
#     path2 = tf.keras.layers.Conv1D(filters[1][1], kernel_size=2, activation="relu", padding=PADDING)(path2)

#     # 3 kernel
#     path3 = tf.keras.layers.Conv1D(filters[2][0], kernel_size=1, activation="relu", padding=PADDING)(layer)
#     path3 = tf.keras.layers.Conv1D(filters[2][1], kernel_size=3, activation="relu", padding=PADDING)(path3)

#     # 4 kernel
#     path4 = tf.keras.layers.Conv1D(filters[3][0], kernel_size=1, activation="relu", padding=PADDING)(layer)
#     path4 = tf.keras.layers.Conv1D(filters[3][1], kernel_size=4, activation="relu", padding=PADDING)(path4)

#     # 5 kernel
#     path5 = tf.keras.layers.Conv1D(filters[4][0], kernel_size=1, activation="relu", padding=PADDING)(layer)
#     path5 = tf.keras.layers.Conv1D(filters[4][1], kernel_size=5, activation="relu", padding=PADDING)(path5)

#     return tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3, path4])


# def get_model():
#     input_layer = tf.keras.layers.DenseFeatures(numeric_columns)
#     feature_layer = input_layer(feature_layer_inputs)

#     layer = tf.keras.layers.Reshape((RECORD_LENGTH - 1, 1))(feature_layer)
#     layer = InceptionLayer(layer, [256, (128, 64), (128, 32), (128, 16), (128, 8)])
#     # layer = InceptionLayer(layer, [32, (128, 64), (128, 32), (128, 16), (128, 8)])

#     # Flatten
#     layer = tf.keras.layers.Flatten()(layer)

#     # Dense
#     layer = tf.keras.layers.Dense(units=512)(layer)
#     layer = tf.keras.layers.Dense(units=128)(layer)
#     layer = tf.keras.layers.Dense(units=64)(layer)

#     layer = tf.keras.layers.Dense(units=1)(layer)
#     model = tf.keras.models.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=layer)
#     return model

# model = get_model()
# model.summary()

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     optimizer="adam",
#     metrics=["accuracy"]
# )

# train_data = packed_train_data.shuffle(250)
# test_data = packed_test_data

# model.fit(train_data, epochs=100)

# test_loss, test_accuracy = model.evaluate(test_data)

# print("\n\nTest loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_accuracy))

# predictions = model.predict(test_data)

# # Show some results
# for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
#   prediction = tf.sigmoid(prediction).numpy()
#   print("Predicted being real glucose: {:.2%}".format(prediction[0]),
#         " | Actual outcome: ",
#         ("REAL" if bool(survived) else "ADDITIONAL"))

# model.save_weights("./checkpoints/cnn-inceptions")