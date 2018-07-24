import collections
import csv
import datetime
from sklearn.cross_validation import train_test_split

import numpy as np

SCHEMA_DESC = collections.OrderedDict([
    ('datetime', str),
    ('season', float),
    ('holiday', float),
    ('workingday', float),
    ('weather', float),
    ('temp', float),
    ('atemp', float),
    ('humidity', float),
    ('windspeed', float),
    ('casual', float),
    ('registered', float),
    ('count', float)
])


def process_data_row(row: list, schema: list) -> list:
    processed = []

    if len(row) != len(schema):
        return processed

    for idx, value in enumerate(row):
        if schema[idx]:
            processed.append(schema[idx](row[idx]))

    return processed


def read_csv(file: str) -> np.array:
    with open(file) as csv_file:
        reader = csv.reader(csv_file)
        data = []

        for idx, row in enumerate(reader):
            if not idx:
                continue

            data.append(process_data_row(row, [SCHEMA_DESC[c] for c in SCHEMA_DESC]))

    return np.array(data)


def write_csv(file: str, data: np.array, header: list) -> None:
    with open(file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        if header:
            writer.writerow(header)

        writer.writerows(data)


def remove_count_outliers(data: np.array, std_margin: float):
    return data[np.abs(data[:, 11].astype(np.float) - data[:, 11].astype(np.float).mean())
                <= std_margin * data[:, 11].astype(np.float).std()]


def extract_date_time(data: np.array, schema: dict) -> (np.array, list):
    extracted_dates = []

    for r in data:
        parsed_date = datetime.datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S")
        date = [parsed_date.month, parsed_date.weekday(), parsed_date.hour]
        extracted_dates.append(date)

    header = ['month', 'week_day', 'hour']
    header += [v for v in schema.keys() if v != 'datetime']

    return np.c_[np.array(extracted_dates), data[:, 1:]], header


def remove_column(data: np.array, index: int, header: list):
    data = np.delete(data, index, 1)
    del header[index]

    return data, header


def load_data_set(file: str) -> np.array:
    return read_csv(file)


def round_numbers(data, columnNo):
    for r in data:
        r[columnNo] = round(float(r[columnNo]))
    return data


def remove_humidity(data, humi):
    return data[np.abs(data[:, 9].astype(np.float)) != humi]


def remove_weather(data, weather):
    return data[np.abs(data[:, 6].astype(np.float)) != weather]


def remove_records_before_date(data, limit_date):
    limit_date_typed = datetime.datetime.strptime(limit_date, "%Y-%m-%d %H:%M:%S")
    for r in data:
        parsed_date = datetime.datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S")

        if parsed_date < limit_date_typed:
            r[0] = limit_date

    return data[(data[:, 0].astype(np.str)) != limit_date]


def prepare_data_set(data: np.array) -> (np.array, list):
    data = remove_records_before_date(data, '2011-04-01 00:00:00')
    data_processed, header = extract_date_time(data, SCHEMA_DESC)

    data_processed = round_numbers(data_processed, 7)
    data_processed = round_numbers(data_processed, 8)
    data_processed = round_numbers(data_processed, 9)
    data_processed = round_numbers(data_processed, 10)

    data_processed, header = remove_column(data_processed, 12, header)
    data_processed, header = remove_column(data_processed, 11, header)

    data_processed = remove_count_outliers(data_processed, 2.5)

    data_processed = remove_humidity(data_processed, 0)
    data_processed = remove_weather(data_processed, 4)

    return remove_column(data_processed, 8, header)


def divide_dataset(sliced_data, header) -> (np.array, np.array, np.array, np.array):
    data_train, data_test, target_train, target_test = train_test_split(sliced_data, header, test_size=0.2)
    write_csv('test_data.csv', target_test, 'test data')
    return data_train, data_test, target_train, target_test


def main():
    data = load_data_set('train.csv')
    processed, header = prepare_data_set(data)

    write_csv('train_processed.csv', processed, header)


main()
