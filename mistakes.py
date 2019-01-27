from __future__ import print_function

# keras imports
import keras
from keras.models import load_model
from keras import backend as K

# python imports
import cv2
import numpy as np

model = load_model("models/model.h5")


# input image dimensions
num_classes = 2
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
        if not (emotion == '6' or emotion == '3'):
            continue
        if emotion == '6':
            emotion = 0
        if emotion == '3':
            emotion = 1

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
        y.append(emotion)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_dev = np.array(x_dev)
y_dev = np.array(y_dev)
x_test = np.array(x_test)
y_test = np.array(y_test)

# flexibility in case of different backends (support different backends)
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


y_test_predict = np.argmax(model.predict(x_test), axis=1)

wrong_neutral = np.where((y_test - y_test_predict) == -1)[0]
wrong_happiness = np.where((y_test - y_test_predict) == 1)[0]

for index in wrong_neutral:
    image = x_test[index].reshape(48, 48)
    image = (image * 255).astype(np.int8)
    cv2.imwrite('wn/%s.jpg' % index, image)

for index in wrong_happiness:
    image = x_test[index].reshape(48, 48)
    image = (image * 255).astype(np.int8)
    cv2.imwrite('wh/%s.jpg' % index, image)

print(wrong_neutral)
print(wrong_happiness)
