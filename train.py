from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np


batch_size = 128
num_classes = 2
epochs = 50

# input image dimensions
img_rows, img_cols = 48, 48

# the data, shuffled and split between train and test sets
x_train = []
y_train = []
x_dev= []
y_dev = []
x_test = []
y_test = []

with open('./data/fer2013.csv') as file:
    lines = file.read().split('\n')[1:]
    lines.remove('')

    for line in lines:
        emotion, pixels, usage = line.split(',')
        if not (emotion == '0' or emotion == '1'):
            continue

        if usage == 'Training':
            x, y = x_train, y_train
        elif usage == 'PrivateTest':
            x, y = x_test, y_test
        elif usage == 'PublicTest':
            x, y = x_dev, y_dev 
        else:
            continue

        pixels = [int(p) for p in pixels.split(' ')]
        x.append(np.array(pixels))
        y.append(int(emotion))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_dev = np.array(x_dev)
y_dev = np.array(y_dev)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Flexibility in case of different backends (support different backend)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_dev = x_dev.reshape(x_dev.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_dev = x_dev.reshape(x_dev.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_dev = x_dev.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_dev /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_dev.shape[0], 'dev samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_dev = keras.utils.to_categorical(y_dev, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# for w in model.get_weights():
#     print(w.shape)

checkpointer = keras.callbacks.ModelCheckpoint(
    'models/model.h5',
    save_best_only = True,
    monitor = 'val_acc',
    mode = 'auto',
    verbose = 1
)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[checkpointer],
          validation_data=(x_dev, y_dev)
          )

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])