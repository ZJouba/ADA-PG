from importer import importCSV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sea
import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend

def getData():
    from keras.datasets import mnist
    train_data, predict_data = importCSV()

    y_train = train_data.iloc[:, 0].values.astype(np.int32)
    X_train = train_data.iloc[:, 1:].values

    X_predict = predict_data.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float64)) 
    X_predict = scaler.fit_transform(X_predict.astype(np.float64))

    global encoder 
    encoder = OneHotEncoder(categories='auto', sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1,1))

    X_train = X_train.reshape( -1, 28, 28, 1)
    X_predict = X_predict.reshape( -1, 28, 28, 1)

    return X_train, X_predict, y_train


def trainModel(input_shape, X_train, y_train, epochs, model_path, batch_size, early_stopping_epochs):
    from keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, Lambda
    from keras.layers.advanced_activations import PReLU
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import RMSprop, Adam
    import keras as k

    es = EarlyStopping(monitor='loss', verbose=1, patience=early_stopping_epochs)
    mc = ModelCheckpoint(model_path, monitor='loss', save_best_only=True)

    generator = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.3,
    )

    model = Sequential()

    model.add(Conv2D(32, (4,4), strides=(2,2), input_shape=(28,28,1)))
    model.add(PReLU())
    model.add(Dropout(rate=0.4))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(32, pool_size=(3,3)))

    model.add(Conv2D(64, (4,4), strides=(2,2), input_shape=(28,28,1)))
    model.add(PReLU())
    model.add(Dropout(rate=0.4))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(32, pool_size=(3,3)))

    model.add(Conv2D(128, (3,3), strides=(1,1), input_shape=(28,28,1)))
    model.add(PReLU())
    model.add(Dropout(rate=0.4))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(32, pool_size=(3,3)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(PReLU())
    model.add(Dropout(rate=0.2))
    model.add(BatchNormalization())

    model.add(Dense(256))
    model.add(PReLU())
    model.add(Dropout(rate=0.2))
    model.add(BatchNormalization())

    model.add(Dense(128))
    model.add(PReLU())
    model.add(Dropout(rate=0.2))
    model.add(BatchNormalization())

    model.add(Dense(10,activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    logging = model.fit_generator(
        generator.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch= X_train.shape[0],
        callbacks=[es, mc],
        shuffle=True,
        )

    pickle.dump(model, open(model_path, 'wb'))

    return model


def main(epochs, batch_size, early_stopping_epochs):
    current_path = os.path.dirname(__file__)
    model_path = os.path.join(current_path, 'trained_model.p')

    X_train, X_predict, y_train = getData()
    
    input_shape = (28, 28, 1)

    if os.path.isfile(model_path):
        if input('Model already trained. Retrain [y/n] \t') == 'y':

            model = trainModel(
                input_shape, 
                X_train, 
                y_train,
                epochs, 
                model_path, 
                batch_size,
                early_stopping_epochs
                )
        else:
            model = pickle.load(open(model_path, 'rb'))
    else:
        model = trainModel(
            input_shape, 
            X_train, 
            y_train,
            epochs, 
            model_path, 
            batch_size,
            early_stopping_epochs
            )
    
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


EPOCHS = 50
BATCH_SIZE = 64
EARLY_STOPPING_EPOCHS = 4 # NUMBER OF EPOCHS TO CONSIDER BEFORE TERMINATING

main(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping_epochs=EARLY_STOPPING_EPOCHS,
    )
