# -*- coding: utf-8 -*-
# Usage: python keras_weights_converter.py <path to keras weights (.h5)> <path to output weights in text format (.txt)>
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import sys
os.environ["THEANO_FLAGS"] = "floatX=float32,device=cpu,force_device=True"
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


np.random.seed(2016)


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
    if 0:
        model2.add(ZeroPadding2D((1, 1)))
        model2.add(Convolution2D(64, 3, 3, activation='relu', weights=model.layers[3].get_weights()))
        model2.add(MaxPooling2D((2, 2), strides=(2, 2)))
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

    return model, model2


def create_weights_text_file(model, out_file):
    weights = dict()
    bias = dict()
    np.set_printoptions(precision=18)
    out = open(out_file, "w")
    weights[0], bias[0] = model.layers[1].get_weights()
    weights[1], bias[1] = model.layers[3].get_weights()
    weights[2], bias[2] = model.layers[6].get_weights()
    weights[3], bias[3] = model.layers[8].get_weights()
    weights[4], bias[4] = model.layers[11].get_weights()
    weights[5], bias[5] = model.layers[13].get_weights()
    weights[6], bias[6] = model.layers[15].get_weights()
    weights[7], bias[7] = model.layers[18].get_weights()
    weights[8], bias[8] = model.layers[20].get_weights()
    weights[9], bias[9] = model.layers[22].get_weights()
    weights[10], bias[10] = model.layers[25].get_weights()
    weights[11], bias[11] = model.layers[27].get_weights()
    weights[12], bias[12] = model.layers[29].get_weights()
    weights[13], bias[13] = model.layers[32].get_weights()
    weights[14], bias[14] = model.layers[34].get_weights()
    weights[15], bias[15] = model.layers[36].get_weights()


    for z in range(13):
        print('Shape weights {}: {}'.format(z, weights[z].shape))
        for i in range(weights[z].shape[0]):
            for j in range(weights[z].shape[1]):
                for k in range(weights[z].shape[2]):
                    for l in range(weights[z].shape[3]):
                        out.write(str(weights[z][i, j, k, l].astype(np.float64)) + " ")
        out.write("\n")
        print('Shape bias {}: {}'.format(z, bias[z].shape))
        for i in range(bias[z].shape[0]):
            out.write(str(bias[z][i].astype(np.float64)) + " ")
        out.write("\n")

    for z in range(13, 16):
        print('Shape weights {}: {}'.format(z, weights[z].shape))
        for i in range(weights[z].shape[0]):
            for j in range(weights[z].shape[1]):
                out.write(str(weights[z][i, j].astype(np.float64)) + " ")
        out.write("\n")
        print('Shape bias {}: {}'.format(z, bias[z].shape))
        for i in range(bias[z].shape[0]):
            out.write(str(bias[z][i].astype(np.float64)) + " ")
        out.write("\n")

    out.close()


if __name__ == '__main__':
    print('Read model...')
    if len(sys.argv) != 3:
        print('Usage: python keras_weights_converter.py <path to keras weights (.h5)> <path to output weights in text format (.txt)>')
    else:
        model, model_checker = VGG_16(sys.argv[1])
        print(model.summary())
        create_weights_text_file(model, sys.argv[2])
        print('Complete!')
