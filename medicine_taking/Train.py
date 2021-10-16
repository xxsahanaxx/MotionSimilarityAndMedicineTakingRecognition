import pandas as pd
import numpy as np
import os, time, cv2, tqdm, datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# your code here
with tf.device('/gpu:0'):
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from warnings import filterwarnings
from keras.preprocessing.image import ImageDataGenerator
# Compile the model
from keras.callbacks import TensorBoard

from keras.models import Model
import keras.layers as L


import re

filterwarnings('ignore')

SIZE = (224,224)
POSITIVES_PATH_TRAIN = 'data/Train/Class1/'
NEGATIVES_PATH_TRAIN = 'data/Train/Class2/'

POSITIVES_PATH_VALID = 'data/Val/Class1/'
NEGATIVES_PATH_VALID = 'data/Val/Class2/'

# POSITIVES_PATH_TEST =
# NEGATIVES_PATH_TEST =




def data_to_lstm_format(POSITIVES_PATH, NEGATIVES_PATH, look_back = 4):
    data = np.array([])
    labels = np.array([])
    numbers = []
    # POSITIVE LABELS
    for value in os.listdir(POSITIVES_PATH):
        numbers.append(int(re.findall(r'\d+', value.split('_')[2])[0]))

    # filter by video
    for numb in np.unique(numbers):
        frames = []
        # append image name
        for value in os.listdir(POSITIVES_PATH):
            if int(re.findall(r'\d+', value.split('_')[2])[0]) == numb:
                frames.append(value)
        # sort image frame by frame number
        frames = sorted(frames, key = lambda x: int(re.findall(r'\d+', x.split('_')[-1].split('.')[0])[0]))
        image_data = np.zeros((len(frames), 1024))

        # get feature vector from vgg16 for each frame and stack
        for index, image in enumerate(frames):
            img = cv2.imread(POSITIVES_PATH + image)
            vect = feat_extractor.predict(img.reshape(1,224,224,3))
            image_data[index,:] = vect

        # for each frame get tensor with lookbacks
        stacked_data = np.zeros((len(frames), look_back, 1024))
        for index in range(len(frames)):
            labels = np.append(labels, [1])
            stacked_data[index, 0, :] = image_data[index]
            for lb in range(1, look_back):
                if index - lb >= 0:
                    stacked_data[index, lb, :] = image_data[index - lb]
                else:
                    stacked_data[index, lb, :] = np.zeros(1024)

        if data.shape[0] == 0:
            data = stacked_data
        else:
            data = np.concatenate([data, stacked_data], axis = 0)

    for value in os.listdir(NEGATIVES_PATH):
        numbers.append(int(re.findall(r'\d+', value.split('_')[2])[0]))

    # filter by video
    for numb in np.unique(numbers):
        frames = []
        # append image name
        for value in os.listdir(NEGATIVES_PATH):
            if int(re.findall(r'\d+', value.split('_')[2])[0]) == numb:
                frames.append(value)
        # sort image frame by frame number
        frames = sorted(frames, key = lambda x: int(re.findall(r'\d+', x.split('_')[-1].split('.')[0])[0]))
        image_data = np.zeros((len(frames), 1024))

        # get feature vector from vgg16 for each frame and stack
        for index, image in enumerate(frames):
            img = cv2.imread(NEGATIVES_PATH + image)
            vect = feat_extractor.predict(img.reshape(1,224,224,3))
            image_data[index,:] = vect

        # for each frame get tensor with lookbacks
        stacked_data = np.zeros((len(frames), look_back, 1024))
        for index in range(len(frames)):
            labels = np.append(labels, [0])
            stacked_data[index, 0, :] = image_data[index]
            for lb in range(1, look_back):
                if index - lb >= 0:
                    stacked_data[index, lb, :] = image_data[index - lb]
                else:
                    stacked_data[index, lb, :] = np.zeros(1024)

        if data.shape[0] == 0:
            data = stacked_data
        else:
            data = np.concatenate([data, stacked_data], axis = 0)

    # one hot labels
    from keras.utils import to_categorical
    labels = to_categorical(labels)
    return data, labels

    

# Create the model
def build_feat_extractor():
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(2, activation='softmax'))
    return model
    
def build_model():
    inp = L.Input(shape = (LOOK_BACK, num_features))
    
    """ Use CuDNNLSTM if your machine supports CUDA
        Training time is significantly faster compared to LSTM """
    
    #x = L.LSTM(64, return_sequences = True)(inp)
    x = L.CuDNNLSTM(64, return_sequences = True)(inp)
    x = L.Dropout(0.2)(x)
    #x = L.LSTM(16)(x)
    x = L.CuDNNLSTM(16)(x)
    out = L.Dense(2, activation = 'softmax')(x)
    model = Model(inputs = [inp], outputs = [out])
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    return model
    
    
if __name__ == '__main__':
    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE[0], SIZE[1], 3))

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False
    
    #   labels enabled for fine-tuning
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    build_feat_extractor().summary()
    train_batchsize = 64
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory('data/Train/', class_mode='categorical', batch_size=train_batchsize, target_size = SIZE)

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_generator = val_datagen.flow_from_directory('data/Val/', class_mode='categorical', batch_size=train_batchsize, target_size = SIZE)

    model = build_feat_extractor()
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

    # Train the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        validation_data = val_generator,
        validation_steps = val_generator.samples/val_generator.batch_size,
        epochs=10,
        verbose=2)
 
    # Save the trained model to disk
    model.save('weights/Feature_Extractor.h5')



    inp = model.input
    out = model.layers[-4].output
    feat_extractor = Model(inputs = [inp], outputs = [out])
    feat_extractor.summary()

    feat_extractor.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
              
              
    LOOK_BACK = 4

    log_dir = "data/_training_logs/rnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

    model = build_model()
    history = model.fit(tr_data, tr_labels,
                    validation_data = (val_data, val_labels),
                    callbacks = [tensorboard_callback],
                    verbose = 2, epochs = 20, batch_size = 64)
    model.save('weights/RNN.h5')
