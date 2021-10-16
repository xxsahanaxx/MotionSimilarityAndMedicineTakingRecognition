
"""
This script is responsible for gathering the frames of every video, 
collating them into a "sequence" and labelling every sequence by 
the name of the action being performed, to store them in train and
test data subsets. 

"""
import folder_setup
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import numpy as np


# Create a dictionary for each label and index
label_map = {label:num for num, label in enumerate(folder_setup.exercises)}
print(label_map)

'''
This function preprocesses data from the dataset by video 
sequences and their respective labels and packs them into 
a particular directory specified by the user. 
    Input: loc: location to save the preprocessed data to

'''
def preprocess_frames(loc):
    # Create sequences of 30 frames (video) and labels for each of the 30 frames
    sequences, labels = [], [] # should become the length of number of videos captured 
    for exercise in folder_setup.exercises:
        print(exercise)
        for sequence in range(1,(folder_setup.no_videos*3)+1):
            #print("seq:",sequence)
            video = []
            for frame_num in range(folder_setup.sequence_length):
                #print("f_n:",frame_num)
                res = np.load(folder_setup.os.path.join(folder_setup.DATA_PATH, exercise, str(sequence), "{}.npy".format(frame_num)))
                video.append(res)
            sequences.append(video)
            labels.append(label_map[exercise])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    np.save(os.path.join(loc, "res", "X_five"), arr=X)
    np.save(os.path.join(loc, "res", "y_five"), arr=y)

'''
This function loads data from the location/directory specified 
by the user. Test data contains 8% of the samples. 
    Input: loc: location to load the preprocessed data from
    Outputs: X_train: video sequence set (92% of frames)
             X_test: video sequence set (8% of frames)
             y_train: label set (92% of frames)
             y_test: label set (8% of frames)
'''
def load_train_data(loc):
    # Load numpy arrays
    X = np.load(os.path.join(loc,"res","X_five.npy"))
    y = np.load(os.path.join(loc,"res","y_five.npy"))

    # Train-test split
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08)
    #print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_frames(os.getcwd())