import csv
import os

DATASET_FOLDER = os.path.join('..', 'data', 'twitter')
POSITIVE_FILE_NAME = 'positive.csv'
NEGATIVE_FILE_NAME = 'negative.csv'
DATASET_PERCENT_SPLIT = 0.80


def load(file_name):
    arr = []
    with open(os.path.join(DATASET_FOLDER, file_name), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        for ind, text in enumerate(reader):
            arr.append(text[3])
            print(ind)
    return arr


def save_arr(arr, folder, category):
    folder = os.path.join(DATASET_FOLDER, folder, category)
    for ind, content in enumerate(arr):
        with open(os.path.join(folder, str(ind) + ".txt"), 'w') as f:
            f.write(content)


def create_dataset(file_name, category):
    arr = load(file_name)
    size = len(arr)
    train_size = int(size * DATASET_PERCENT_SPLIT)
    training, test = arr[:train_size], arr[train_size:]
    save_arr(training, "train", category)
    save_arr(test, "test", category)


def main():
    create_dataset(POSITIVE_FILE_NAME, "pos")
    create_dataset(NEGATIVE_FILE_NAME, "neg")


if __name__ == "__main__":
    main()
