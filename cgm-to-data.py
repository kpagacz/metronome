import csv
import numpy as np 

from config import RECORDS, MAX_ADDITIONAL_RECORDS, INTERVAL, file_name, RECORD_LENGTH

with open(file_name, "r", newline="") as source:
    reader = csv.reader(source)
    last_row = next(reader)
    with open("data.csv", "w", newline="") as destination:
        writer = csv.writer(destination)
        batch = []
        for i in range(RECORD_LENGTH):
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

