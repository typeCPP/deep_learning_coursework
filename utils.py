import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.callbacks import LearningRateScheduler


def prepare_data():
    # Opening file for reading in binary mode
    with open('input/traffic-signs-preprocessed/data2.pickle', 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # dictionary type

    # Preparing y_train and y_validation for using in Keras
    data['y_train'] = to_categorical(data['y_train'], num_classes=43)
    data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)

    # Making channels come at the end
    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)
    return data


def compile_model(kernel_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=kernel_size, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(43, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_model(model, data):
    epochs = 5
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))

    model.fit(data['x_train'], data['y_train'],
              batch_size=128, epochs=epochs,
              validation_data=(data['x_validation'], data['y_validation']),
              callbacks=[annealer], verbose=1)
    return model
