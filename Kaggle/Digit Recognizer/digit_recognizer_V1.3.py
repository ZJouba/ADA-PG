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

def getData(validation_set_size):
    train_data, predict_data = importCSV(None)

    num_instances = len(train_data)
    valid_size = round((validation_set_size/100) * num_instances)
    test_size = num_instances - valid_size

    y_train = train_data.iloc[:test_size, 0].values.astype(np.int32)
    X_train = train_data.iloc[:test_size, 1:].values 

    y_valid = train_data.iloc[test_size:test_size+valid_size, 0].values.astype(np.int32)
    X_valid = train_data.iloc[test_size:test_size+valid_size, 1:].values 
    X_predict = predict_data.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float64))
    X_valid = scaler.fit_transform(X_valid.astype(np.float64))    
    X_predict = scaler.fit_transform(X_predict.astype(np.float64))

    global encoder 
    encoder = OneHotEncoder(categories='auto', sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1,1))
    y_valid = encoder.fit_transform(y_valid.reshape(-1,1))

    X_train = X_train.reshape( -1, 28, 28, 1)
    X_valid = X_valid.reshape( -1, 28, 28, 1)
    X_predict = X_predict.reshape( -1, 28, 28, 1)

    return X_train, X_valid, X_predict, y_train, y_valid


def trainModel(input_shape, X_train, y_train, X_valid, y_valid, epochs, model_path, batch_size, early_stopping_epochs):
    from keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D, Lambda
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import RMSprop, Adam
    import keras as k

    es = EarlyStopping(monitor='val_acc', verbose=1, patience=early_stopping_epochs)
    mc = ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True)

    generator = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.3,
    )

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    logging = model.fit_generator(
        generator.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch= X_train.shape[0],
        validation_data=(X_valid, y_valid),
        callbacks=[es, mc],
        shuffle=True,
        )

    pickle.dump(model, open(model_path, 'wb'))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(logging.history['loss'], label='Training loss')
    ax[0].plot(logging.history['val_loss'], label='Validation loss', axes=ax[0])
    ax[0].legend(loc='best')
    ax[1].plot(logging.history['acc'], label='Training accuracy')
    ax[1].plot(logging.history['val_acc'], label='Validation accuracy', axes=ax[1])
    ax[1].legend(loc='best')
    plt.show()

    return model


def main(epochs, validation_set_size, batch_size, early_stopping_epochs):
    current_path = os.path.dirname(__file__)
    model_path = os.path.join(current_path, 'trained_model.p')

    X_train, X_valid, X_predict, y_train, y_valid = getData(
        validation_set_size)
    
    input_shape = (28, 28, 1)

    if os.path.isfile(model_path):
        if input('Model already trained. Retrain [y/n] \t') == 'y':

            model = trainModel(
                input_shape, 
                X_train, 
                y_train, 
                X_valid, 
                y_valid, 
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
            X_valid, 
            y_valid, 
            epochs, 
            model_path, 
            batch_size,
            early_stopping_epochs
            )
    
    print(model.evaluate(X_valid, y_valid, batch_size=batch_size))

    matrix_pred = model.predict(X_valid, batch_size=batch_size,verbose=1).argmax(axis=1)
    conf_matrix = confusion_matrix(encoder.inverse_transform(y_valid), matrix_pred)
    np.fill_diagonal(conf_matrix, 0)
    conf_matrix = pd.DataFrame(conf_matrix, index=[i for i in range(0,10)], columns=[i for i in range(0,10)])
    plt.figure()
    sea.heatmap(conf_matrix, annot=True, fmt='g', cmap=sea.cm.rocket)
    plt.pause(1)

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
VALIDATION_SET_SIZE = 0.2  # PERCENTAGE
BATCH_SIZE = 64
EARLY_STOPPING_EPOCHS = 2 # NUMBER OF EPOCHS TO CONSIDER BEFORE TERMINATING

main(
    epochs=EPOCHS,
    validation_set_size=VALIDATION_SET_SIZE,
    batch_size=BATCH_SIZE,
    early_stopping_epochs=EARLY_STOPPING_EPOCHS,
    )
