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
import seaborn as sea

plt.ion()

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

    X_train = np.pad(X_train.reshape( -1, 28, 28, 1), ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_valid = np.pad(X_valid.reshape( -1, 28, 28, 1), ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_predict = np.pad(X_predict.reshape( -1, 28, 28, 1), ((0,0),(2,2),(2,2),(0,0)), 'constant')

    return X_train, X_valid, X_predict, y_train, y_valid


def trainModel(input_shape, X_train, y_train, X_valid, y_valid, epochs, model_path, batch_size, early_stopping_epochs):
    import plaidml.keras
    plaidml.keras.install_backend()
    import plaidml.keras.backend
    from keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import RMSprop, Adam
    import keras as k

    es = EarlyStopping(monitor='val_acc', verbose=1, patience=early_stopping_epochs)
    mc = ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True)

    # model = Sequential()
    # # LAYER
    # model.add(Conv2D(
    #     20,
    #     kernel_size=(4, 4),
    #     input_shape=input_shape,
    #     activation='tanh',
    #     padding='same',
    #     ))

    # # LAYER
    # # model.add(BatchNormalization())

    # # LAYER
    # # model.add(Conv2D(
    # #     64,
    # #     kernel_size=(5, 5),
    # #     activation='tanh',
    # #     padding='valid',
    # #     ))

    # # LAYER
    # # model.add(BatchNormalization())

    # # LAYER
    # model.add(MaxPooling2D(
    #     pool_size=(2, 2),
    #     ))

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.25
    # #     ))

    # # LAYER
    # model.add(Conv2D(
    #     40,
    #     kernel_size=(5, 5),
    #     activation='tanh',
    #     padding='same',
    #     ))

    # # LAYER
    # # model.add(BatchNormalization())

    # # LAYER
    # # model.add(Conv2D(
    # #     128,
    # #     kernel_size=(5, 5),
    # #     activation='tanh',
    # #     padding='valid',
    # #     ))

    # # LAYER
    # # model.add(BatchNormalization())

    # # LAYER
    # model.add(MaxPooling2D(
    #     pool_size=(3, 3),
    #     ))

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.25
    # #     ))

    # # LAYER
    # # model.add(Conv2D(
    # #     256,
    # #     kernel_size=(3, 3),
    # #     activation='tanh',
    # #     padding='same',
    # #     ))

    # # LAYER
    # # model.add(BatchNormalization())

    # # LAYER
    # # model.add(Conv2D(
    # #     256,
    # #     kernel_size=(3, 3),
    # #     activation='tanh',
    # #     padding='valid',
    # #     ))

    # # LAYER
    # # model.add(BatchNormalization())

    # # LAYER
    # # model.add(MaxPooling2D(
    #     # pool_size=(2, 2),
    #     # ))

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.25
    # #     ))

    # # LAYER
    # # model.add(Conv2D(
    # #     512,
    # #     kernel_size=(3, 3),
    # #     activation='tanh',
    # #     padding='same',
    # #     ))

    # # LAYER
    # # model.add(BatchNormalization())

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.25
    # #     ))

    # # LAYER
    # model.add(Flatten())

    # # LAYER
    # # model.add(Dense(
    # #     1024, 
    # #     activation='relu'
    # #     ))

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.25
    # #     ))

    # # LAYER
    # model.add(Dense(
    #     units=150,
    #     activation='relu',
    #     ))

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.25
    # #     ))

    # # LAYER
    # # model.add(Dense(
    # #     units=256,
    # #     activation='relu',
    # #     ))

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.5
    # #     ))

    # # LAYER
    # # model.add(Dense(
    # #     units=128,
    # #     activation='relu',
    # #     ))

    # # LAYER
    # # model.add(Dropout(
    # #     rate=0.5
    # #     ))

    # # LAYER
    # model.add(Dense(
    #     units=10, 
    #     activation='softmax'
    #     ))

    # print(model.summary())

    # model.compile(
    #     optimizer=Adam(),
    #     loss=k.losses.categorical_crossentropy,
    #     metrics=['accuracy'],
    # )

    model = Sequential()
    model.add(Conv2D(16, kernel_size = (3,3), input_shape=input_shape, padding = 'same'))
    model.add(Conv2D(16, kernel_size = (3,3), padding = 'valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(16, kernel_size = (3,3), padding = 'same'))
    model.add(Conv2D(16, kernel_size = (3,3), padding = 'valid'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    generator = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.3,
    )

    generator.fit(X_train)

    logging = model.fit_generator(
        generator.flow(X_train, y_train, batch_size=10),
        epochs=epochs,
        steps_per_epoch= 10, #X_train.shape[0] // 10,
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

    # x=X_train, y=y_train, batch_size=128, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[es, mc]

    return model


def main(epochs, validation_set_size, batch_size, early_stopping_epochs):
    current_path = os.path.dirname(__file__)
    model_path = os.path.join(current_path, 'trained_model.p')

    X_train, X_valid, X_predict, y_train, y_valid = getData(
        validation_set_size)
    
    input_shape = (32, 32, 1)

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
EARLY_STOPPING_EPOCHS = 50 # NUMBER OF EPOCHS TO CONSIDER BEFORE TERMINATING

main(
    epochs=EPOCHS,
    validation_set_size=VALIDATION_SET_SIZE,
    batch_size=BATCH_SIZE,
    early_stopping_epochs=EARLY_STOPPING_EPOCHS,
    )
