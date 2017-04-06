# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
# os.environ["THEANO_FLAGS"] = "floatX=float32,device=cpu,force_device=True"
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
np.random.seed(2016)


def normalize_image_vgg16(img):
    img[:, 0, :, :] -= 103.939
    img[:, 1, :, :] -= 116.779
    img[:, 2, :, :] -= 123.68
    return img


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    model2 = Sequential()
    model2.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model2.add(Convolution2D(64, 3, 3, activation='relu', weights=model.layers[1].get_weights()))
    model2.add(ZeroPadding2D((1, 1)))
    model2.add(Convolution2D(64, 3, 3, activation='relu', weights=model.layers[3].get_weights()))
    model2.add(MaxPooling2D((2, 2), strides=(2, 2)))
    if 0:
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(128, 3, 3, activation='relu', weights=model.layers[6].get_weights()))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(128, 3, 3, activation='relu', weights=model.layers[8].get_weights()))
        model2.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[11].get_weights()))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[13].get_weights()))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[15].get_weights()))
        model2.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[18].get_weights()))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[20].get_weights()))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[22].get_weights()))
        model2.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[25].get_weights()))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[27].get_weights()))
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[29].get_weights()))
        model2.add(MaxPooling2D((2, 2), strides=(2, 2)))

    return model, model2


def gen_image(in_path):
    img = cv2.imread(in_path)
    if img.shape != (224, 224, 3):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    return img


def dump_debug(data):
    out = open("debug_py.txt", "w")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                out.write("{:.10f}\n".format(data[i][j][k].astype(np.float64)))
    out.close()


if __name__ == '__main__':
    print('Read model...')
    model, model_checker = VGG_16('weights/vgg16_weights.h5')
    im = gen_image("../input/cat_224.png")
    im = np.transpose(im, (2, 0, 1))
    images = np.expand_dims(im, axis=0)
    images = normalize_image_vgg16(images.astype(np.float32))
    print('Classify images...')
    keras_out = model.predict(images)
    print(keras_out[0])
    # dump_debug(keras_out[0])
