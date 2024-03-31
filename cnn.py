import keras.models
import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

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

opt = Adam()

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(43, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 5
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))

h = model.fit(data['x_train'], data['y_train'],
              batch_size=128, epochs=epochs,
              validation_data = (data['x_validation'], data['y_validation']),
              callbacks=[annealer], verbose=1)

keras.models.save_model(model, "model-3x3.h5")

# Compare different filters
# filters = [3, 5, 9, 13, 15, 19, 23, 25, 31]
# model = [0] * len(filters)

# for i in range(len(model)):
#     model[i] = Sequential()
#     model[i].add(Conv2D(32, kernel_size=filters[i], padding='same', activation='relu', input_shape=(32, 32, 3)))
#     model[i].add(MaxPool2D(pool_size=2))
#     model[i].add(Flatten())
#     model[i].add(Dense(500, activation='relu'))
#     model[i].add(Dense(43, activation='softmax'))
#     model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
