"""
This script serves the purpose of defining RNN models for both LSTM and GRU models, 
and allows the user to either train them from scratch or load a model by weights. 
    To train from scratch: set variable 'train' to True
    To load model by weights: set variable 'train' to False

It also implicitly call preprocess function, and saves the features and class labels 
in two numpy arrays, so that the data can be loaded by this script and then trained on,
if training from scratch.

"""

import os
import folder_setup
import preprocess
import test
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, f1_score
import numpy as np
import csv
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Root directory to save models to 
loc = os.getcwd()
terminal = True

# Train state, True (train) or False (load) from terminal command
train = True
# Model type [0] = LSTM, model type [1] = GRU
RNN_type = folder_setup.train_types[0]

'''
This class is responsible for ensuring that training stops
when the loss hits a certain threshold. 
    Input: Callback: import Callback class 

'''
class EarlyStoppingCallback(Callback):
    def on_batch_end(self, batch, logs={}):
        THR = 0.25 # Assign THR with the value at which you want to stop training.
        if logs.get('loss') <= THR:
            self.model.stop_training = True

# Store logs from training to specified directory 
log_dir = os.path.join(loc, 'logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define early stopping callback 
call_stop = EarlyStoppingCallback()

# Check which RNN model is being called, and accordingly ensure the CSV 
# for losses for that model is created
history_logger=CSVLogger('loss'+RNN_type+'.csv', separator=",", append=True)

'''
This function defines the structure of the model, depending on the type of 
model (LSTM/GRU) passed into it. 
    Input: model_type: either "LSTM" or "GRU"
    Output: defined model with relevant architecture
'''
def define_model(model_type):
    if model_type == "LSTM": 
        # Instantiate NN model 
        model = Sequential()
        # Input shape => 258 co-ordinates across 30 frames 
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
        # 2 more LSTM layers with alternating 2 Dropouts 
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dropout(0.2))
        # 3 Fully Connected layers, 1 Dropout after first 
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        # Output classes = final output neurons 
        model.add(Dense(folder_setup.exercises.shape[0], activation='softmax'))
        # Adam optimiser with learning rate 1e-4 and compilation 
        adam = Adam(lr=0.0001)
        # Categorical cross-entropy (because multi-class)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    elif model_type == "GRU":
        # Instantiate NN model
        model = Sequential()
        # Input shape => 258 co-ordinates across 30 frames
        model.add(GRU(units=64, return_sequences=True, activation='relu', input_shape=(30,258)))
        # 2 more GRU layers with alternating 2 Dropouts 
        model.add(GRU(units=128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(units=64, return_sequences=False, activation='relu'))
        model.add(Dropout(0.2))
        # 3 Fully Connected layers, 1 Dropout after first 
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        # Output classes = final output neurons 
        model.add(Dense(folder_setup.exercises.shape[0], activation='softmax'))
        # Adam optimiser with learning rate 1e-4 and compilation 
        adam = Adam(lr=0.0001)
        # Categorical cross-entropy (because multi-class)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

'''
The purpose of this function is to retrieve the preprocessed frames, split the 
dataset into 'train' and 'test' sets and train a defined model on the 'train' set.
    This function is called when the user wants to train a model from scratch. 

    Inputs: model: pre-defined model architecture 
            model_name: name of the 'weights file' for saving purposes
            model_loc: where to save the 'weights file' 
    Output: weighted model
'''
def train_model(model, model_name, model_loc):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = preprocess.load_train_data(loc)
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, call_stop, history_logger]) #TODO: change epochs

    model.save(os.path.join(model_loc, model_name))
    return model

'''
The purpose of this function is to load a weighted model into a pre-defined one. 
    This function is especially useful when the user does not want to train 
    from scratch, and already has some weights saved. 

    Inputs: model: pre-defined model architecture 
            model_name: name of the 'weights file' for load from
            model_loc: where the 'weights file' is located  
    Output: weighted model
'''
def load_model(model, model_name, model_loc):
    model.load_weights(folder_setup.os.path.join(model_loc,model_name))
    print(model.summary())
    return model


if __name__ == "__main__":
    # Upon launch from terminal, declare the RNN model structure 
    model = define_model(RNN_type)

    # Obtain train and test data from split 
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = preprocess.load_train_data(loc)
    print("The training and testing sets are of sizes: ", X_train.shape, X_test.shape)
    
    # Check if the script is set to train from scratch (train = True) 
    if train == True:
        folder_path = folder_setup.os.path.join(folder_setup.os.getcwd(), "res")
        # If the model is an LSTM one 
        if(RNN_type == "LSTM"):
            # Open the respective loss CSV file and clear it 
            open(folder_path+'lossLSTM.csv', mode='w').close()
            model = train_model(model, 'LSTMexercises.h5', os.path.join(loc,"models"))
        # If the model is a GRU one
        elif(RNN_type == "GRU"):
            # Open the respective loss CSV file and clear it 
            open(folder_path+'lossGRU.csv', mode='w').close()
            model = train_model(model, 'GRUexercises.h5', os.path.join(loc,"models"))
    else:
        # If the model is an LSTM one 
        if(RNN_type == "LSTM"):
            model = load_model(model, 'LSTMexercises.h5', os.path.join(loc,"models"))
        # If the model is a GRU one
        elif(RNN_type == "GRU"):
            model = load_model(model, 'GRUexercises.h5', os.path.join(loc,"models"))

    test.test_predictions_on_test_set(loc,model,RNN_type)
