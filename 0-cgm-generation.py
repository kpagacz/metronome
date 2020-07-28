import csv
import numpy as np

from config import RECORDS_TRAIN, RECORDS_TEST, MAX_ADDITIONAL_RECORDS, INTERVAL, file_name

np.random.seed(15)

def generate_data(file_name, record_no):
    with open(file_name, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["TIME", "REAL_VALUE"])
        writer.writerow([0, 1])
        last_real = 0
        for _ in range(record_no):
            additional_records_number = int(np.random.uniform(low=0, high=MAX_ADDITIONAL_RECORDS))
            additional_records_times = np.sort(np.random.uniform(low=5, high=295, size=additional_records_number)) + last_real
            additional_records_labels = [0] * MAX_ADDITIONAL_RECORDS
            additional_records_rows = list(zip(additional_records_times, additional_records_labels))
            writer.writerows(additional_records_rows)
            
            # Next real time
            last_real = last_real + INTERVAL +  \
                int(np.random.uniform(low=-3, high=3))
            writer.writerow([last_real, 1])

generate_data("cgm-train.csv", RECORDS_TRAIN)
generate_data("cgm-test.csv", RECORDS_TEST)