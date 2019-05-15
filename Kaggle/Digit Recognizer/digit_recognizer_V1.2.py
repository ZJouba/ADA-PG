from importer import importCSV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
import seaborn as sea
import xgboost as xgb
from xgboost import XGBClassifier


def getData(validation_set_size):
    train_data, predict_data = importCSV(None)

    num_instances = len(train_data)
    valid_size = round((validation_set_size/100) * num_instances)
    test_size = num_instances - valid_size

    y_train = train_data.iloc[:test_size, 0].values.astype(np.int32)
    X_train = train_data.iloc[:test_size, 1:].values

    y_valid = train_data.iloc[test_size:test_size +
                              valid_size, 0].values.astype(np.int32)
    X_valid = train_data.iloc[test_size:test_size+valid_size, 1:].values
    X_predict = predict_data.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float64))
    X_valid = scaler.fit_transform(X_valid.astype(np.float64))
    X_predict = scaler.fit_transform(X_predict.astype(np.float64))

    return X_train, X_valid, X_predict, y_train, y_valid


def main(epochs, validation_set_size, batch_size, early_stopping_epochs):
    X_train, X_valid, X_predict, y_train, y_valid = getData(
        validation_set_size)

    trainData = xgb.DMatrix(X_train, label=y_train)
    evalData = xgb.DMatrix(X_valid, label=y_valid)

    validationList = [(trainData, 'train'), (evalData, 'validation')]

    params = [("eta", 0.08), ("max_depth", 10), ("subsample", 0.8), ("colsample_bytree", 0.8), (
        "objective", "multi:softmax"), ("eval_metric", "merror"), ("alpha", 8), ("lambda", 2), ("num_class", 10)]

    model = xgb.train(params=params, dtrain=trainData, evals=validationList,
                      verbose_eval=True)
    # model = XGBClassifier()
    # model.fit(
    #     X_train, y_train,
    #     eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #     eval_metric='accuracy',
    #     verbose=True
    # )

    score = model.evals_result()

    print(score)
    sys.exit()

    matrix_pred = model.predict(
        X_valid, batch_size=batch_size, verbose=1).argmax(axis=1)
    conf_matrix = confusion_matrix(
        encoder.inverse_transform(y_valid), matrix_pred)
    np.fill_diagonal(conf_matrix, 0)
    conf_matrix = pd.DataFrame(conf_matrix, index=[i for i in range(
        0, 10)], columns=[i for i in range(0, 10)])
    plt.figure()
    sea.heatmap(conf_matrix, annot=True, fmt='g', cmap=sea.cm.rocket)
    plt.show()

    predictions = model.predict(X_predict, verbose=1).argmax(axis=1)
    predictions = pd.DataFrame(predictions, columns=['Label'])
    predictions.index += 1
    predictions.index.names = ['ImageId']
    predictions.to_csv(os.path.join(current_path, 'submission.csv'))

    if input('\n Submit? [y/n] \t') == 'y':
        import subprocess
        from datetime import datetime

        now = datetime.now()

        subString = 'kaggle competitions submit -c digit-recognizer -f submission.csv -m "' + \
            str(now.strftime("%Y-%m-%d %H:%M") + '"')
        print(subString)
        subprocess.run(subString)


EPOCHS = 500
VALIDATION_SET_SIZE = 20  # PERCENTAGE
BATCH_SIZE = 25
EARLY_STOPPING_EPOCHS = 50  # NUMBER OF EPOCHS TO CONSIDER BEFORE TERMINATING

main(
    epochs=EPOCHS,
    validation_set_size=VALIDATION_SET_SIZE,
    batch_size=BATCH_SIZE,
    early_stopping_epochs=EARLY_STOPPING_EPOCHS,


)
