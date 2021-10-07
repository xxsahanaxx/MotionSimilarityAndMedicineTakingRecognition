""" 
This script sets up the folders for collection of our datasets.
    In the case that these folders already exist, the directory is unaltered. 

If the script is run, the folders are created. 
    Otherwise, its variables are used in other scripts. 
"""

import numpy as np
import os

# Path for exported data, numpy arrays as frames
DATA_PATH = "/Users/sahanasrinivasan/Desktop/COMPSYS700A/actionDetectionTF/dataset"
train_types = ['LSTM','GRU']

# Exercises that we try to detect
# exercises = np.array(['bicepcurls','windwiper','wristrotations'])
exercises = np.array(['drinkingwater', 'heeltoewalking', 'hipstrengthening', 'medicinetaking', 'toeraises'])

# No of videos worth of data
no_videos = 30

# No of frames per video capture
sequence_length = 30

# Folder start
start_folder = 1 
# heel to toe start recording own videos from 52 
# hip strengthening start recording own videos from 39
# toe raises start recording own videos from 20 

'''
This function creates folders in the DATA_PATH directory initialised. If 
the directory has not been created, this function creates folders and 
subfolders inside it.

    The structure is of the form:
    └── keypoint_dataset
        ├── <exercises[0]>
        │   ├── <start_folder>
        │   │   ├── 0.npy
        │   │   ├── ...
        │   │   └── <sequence_length-1>.npy
        │   ├── ...
        │   └── <start_folder+no_videos-1>
        ├── <exercises[1]>
        │   └── ...
        └── <exercises[2]>
            └── ...

If the directories already exist in the DATA_PATH, they are ignored (and
not overwritten on).
'''
def create_folders():

    print("Creating dataset folder in specified directory")
    os.makedirs(DATA_PATH, exist_ok=True)

    print("Creating 'exercise' folders")
    for exercise in exercises: 
        try: 
            # Make directories
            os.makedirs(os.path.join(DATA_PATH, exercise), exist_ok=True)
        except:
            pass
        
        print("Creating 'video' folders")
        for video in range(1,no_videos+1):
            try: 
                # Make directories
                os.makedirs(os.path.join(DATA_PATH, exercise, str(video)), exist_ok=True)
            except:
                pass
    
if __name__ == "__main__":
    create_folders()
