import datetime
import csv


if __name__ == "__main__":
    FILE_NAME = "datasets/brent_crude_oil.csv"
    DATE_FORMAT = "%m/%d/%Y"

    data = csv.reader(open(FILE_NAME, "r"))
    # skip the header
    header = next(data)

    print(f"Number of samples: {sum([1 for row in data])}")

    data = sorted(data, key=lambda x: datetime.datetime.strptime(x[0], DATE_FORMAT).timestamp())

    with open(FILE_NAME, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

