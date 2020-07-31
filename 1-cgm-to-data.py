import csv
import numpy as np 

from config import RECORDS_TRAIN, RECORDS_TEST, MAX_ADDITIONAL_RECORDS, INTERVAL, RECORD_LENGTH

def time_differences(arr):
    for i, time in enumerate(arr[1:]):
        print(i, time)

def window_approach_feature_engineering(file_name_source, file_name_destination):
    with open(file_name_source, "r", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        with open(file_name_destination, "w", newline="") as destination:
            writer = csv.writer(destination)

            # First row - headers - make tf.data creation easier
            var_names = ["var" + str(i) for i in range(RECORD_LENGTH - 1)]
            label_names = ["label" + str(i) for i in range(RECORD_LENGTH)]
            writer.writerow([*var_names, *label_names])

            batch = []
            for _ in range(RECORD_LENGTH):
                batch.append(next(reader))


            for reader_row in reader:
                batch_last_row = batch[0]
                differences = []
                for row in batch[1:]:
                    differences.append(float(row[0]) - float(batch_last_row[0]))
                    batch_last_row = row

                labels = [row[1] for row in batch]
                writer.writerow([*differences, *labels])

                batch.append(reader_row)
                batch = batch[1:]

def lstm_approach(file_name_source, file_name_destination):
    with open(file_name_source, "r", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        with open(file_name_destination, "w", newline="") as destination:
            writer = csv.writer(destination)

            # First row - headers - make tf.data creation easier
            var_names = ["var" + str(i) for i in range(RECORD_LENGTH - 1)]
            writer.writerow([*var_names, "label"])


            batch = []
            for _ in range(RECORD_LENGTH):
                batch.append(next(reader))

            times = list(zip(*batch))
            times_np = np.array([float(value) for value in times[0]])

            differences = np.diff(times_np)

            y_label = times[1][-1]

            writer.writerow([*differences, y_label])
            
            for row in reader:
                batch.append(row)
                batch = batch[1:]
                times = list(zip(*batch))
                times_np = np.array([float(value) for value in times[0]])

                differences = np.diff(times_np)

                y_label = times[1][-1]

                writer.writerow([*differences, y_label])


            


# Window approach
window_approach_feature_engineering("cgm-train.csv", "window-data-train-15.csv")
window_approach_feature_engineering("cgm-test.csv", "window-data-eval-15.csv")

# LSTM approach
lstm_approach("cgm-train.csv", "lstm-data-train-15-variables.csv")
lstm_approach("cgm-test.csv", "lstm-data-eval-15-variables.csv")

