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
            pass


window_approach_feature_engineering("cgm-train.csv", "window-data-train.csv")
window_approach_feature_engineering("cgm-eval.csv", "window-data-eval.csv")



