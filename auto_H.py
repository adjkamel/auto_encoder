import os
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import sys
import time
import warnings
from keras.models import *
from keras.layers import *

from sklearn.utils import shuffle
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from PIL import Image
import time
from keras import backend as K
import tifffile
from datetime import datetime
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import matplotlib

global_option = "testH"
# trainH #contH #testH #trainL #contL #testL #combine
weights = ""


def PrepareData(opt, direction):

    dir_root = "data/"

    if direction == "H":
        trainFolder = "trainH/"
        testFolder = "testH/"
        dir_train = os.listdir(dir_root+"trainH")
        dir_test = os.listdir(dir_root+"testH")

    else:
        trainFolder = "trainL/"
        testFolder = "testL/"
        dir_train = os.listdir(dir_root+"trainL")
        dir_test = os.listdir(dir_root+"testL")

    List_img_train = []
    List_gt_train = []
    List_img_test = []

    if opt == "train":

        for k in range(0, len(dir_train)):
            Xtrain = cv2.imread(dir_root + trainFolder + dir_train[k])
            Xtrain = Xtrain/255
            # print(Xtrain)
            List_img_train.append(Xtrain)
            List_gt_train.append(Xtrain)
        return List_img_train, List_gt_train

    else:

        for k in range(0, len(dir_test)):
            Xtest = cv2.imread(dir_root+testFolder+dir_test[k])
            Xtest = Xtest/255
            List_img_test.append(Xtest)
        return List_img_test


def unet(pretrained_weights=None, input_size=(600, 600, 3)):

    inputs = Input(input_size)

    conv0 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(inputs)
    conv0 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)

    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(pool0)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(pool2)

    up1 = UpSampling2D(size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(up1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(conv4)

    a1 = Add()([conv4, conv2])

    up2 = UpSampling2D(size=(2, 2))(a1)

    conv5 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(up2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(conv5)

    a2 = Add()([conv5, conv1])

    up3 = UpSampling2D(size=(2, 2))(a2)

    conv6 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(up3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(conv6)

    a3 = Add()([conv6, conv0])

    conv5 = Conv2D(3, 3, activation='relu', padding='same',
                   kernel_initializer='he_uniform')(a3)

    model = Model(input=inputs, output=conv5)

    return model


warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')
sess = tf.compat.v1.Session(config=config)
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = '0'
tf.compat.v1.Session(config=config)


if global_option == "trainH":

    model = unet(pretrained_weights=None, input_size=(600, 600, 3))
    model.summary()
    X, Y = PrepareData("train", "H")
    X, Y = np.array(X), np.array(Y)
    print(X.shape, "fffffffffffffffffffffff", Y.shape)
    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss='mean_squared_error', metrics=['accuracy'])
    # model = tf.keras.models.load_model(os.path.abspath("best_model_l8.hdf5"))
    checkpoint = ModelCheckpoint(
        "weights_H.hdf5", monitor='loss', verbose=1, save_best_only=False, mode='max')
    logdir = "logsH" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # hist1 = model.fit(X, Y, batch_size=16, epochs=10, verbose=1, callbacks=[tensorboard_callback])
    #  # validation_data=(X_test,y_test),
    hist1 = model.fit(X, Y, batch_size=1, epochs=2000,
                      verbose=1, callbacks=[checkpoint])
    ### Plot the change in loss over epochs ###
    for key in ['loss']:
        plt.plot(hist1.history['loss'], label=key)
    plt.legend()
    plt.show()


if global_option == "contH":
    #model = unet(pretrained_weights = None,input_size = (600,600,1))
    X, Y = PrepareData("train", "H")
    X, Y = np.array(X), np.array(Y)
    #model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model = tf.keras.models.load_model(os.path.abspath("weights_H.hdf5"))
    K.set_value(model.optimizer.lr, 1e-6)

    checkpoint = ModelCheckpoint(
        "weights_H.hdf5", monitor='loss', verbose=1, save_best_only=False, mode='max')
    logdir = "logsL" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # hist1 = model.fit(X, Y, batch_size=16, epochs=10, verbose=1, callbacks=[tensorboard_callback])
    #  # validation_data=(X_test,y_test),
    hist1 = model.fit(X, Y, batch_size=1, epochs=2000,
                      verbose=1, callbacks=[checkpoint])
    ### Plot the change in loss over epochs ###
    for key in ['loss']:
        plt.plot(hist1.history['loss'], label=key)
    plt.legend()
    plt.show()


if global_option == "testH":

    X = PrepareData("test", "H")

    X = np.array(X)

    model = tf.keras.models.load_model(os.path.abspath("weights_H.hdf5"))
    Xres = model.predict(X)
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',X[0])
    # print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbb',Xres[0])
    for k in range(0, len(X)):
        matplotlib.image.imsave('name.png', Xres[k]/np.max(Xres[k]))

        fig = plt.figure(figsize=(10, 7))

        fig.add_subplot(1, 2, 1)
        plt.imshow(X[k])
        plt.title('original')
        plt.axis('off')

        fig.add_subplot(1, 2, 2)
        plt.imshow(Xres[k])
        plt.title('reconstruction')
        plt.axis('off')

        plt.show()


# if global_option == "trainH":
# if global_option == "contH":
# if global_option == "testH":


'''
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
'''
