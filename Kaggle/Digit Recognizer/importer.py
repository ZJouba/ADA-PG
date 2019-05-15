import os
import pandas as pd


def importCSV(rows):
    current_path = os.path.dirname(__file__)
    train_file = os.path.join(current_path, 'train.csv')
    test_file = os.path.join(current_path, 'test.csv')
    train_data = pd.read_csv(train_file, nrows=rows)
    predict_data = pd.read_csv(test_file)

    return train_data, predict_data
