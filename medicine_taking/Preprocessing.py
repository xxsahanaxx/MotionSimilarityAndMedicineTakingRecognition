import pandas as pd
import numpy as np
import os, time, cv2, tqdm, warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings('ignore')

tqdm.pandas()

TARGET = 'dataset/Class1'
NORMAL_PATH = 'dataset/Class2/'
ORIGINAL_PATH = TARGET + '/'

SIZE = (224, 224)
TRAIN_SIZE = (0,120)
VAL_SIZE = (120,139)
#TEST_SIZE = (0,0)

TAKE_FRAME = 1

def create_dir(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def initialize_dirs():
    create_dir('data/')
    create_dir('data/Train/')
    create_dir('data/Val/')
    create_dir('data/_training_logs/')
    create_dir('data/Train/Class1')
    create_dir('data/Train/Class2')
    create_dir('data/Val/Class1')
    create_dir('data/Val/Class2')
    create_dir('weights')

def get_size(start_path = 'data/'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
    
def generate_data(POSITIVES_PATH, NEGATIVES_PATH, VIDEO_IDX = (0,2)):
    # process original videos
    print('Processing class1 videos...')
    for value in tqdm(os.listdir(ORIGINAL_PATH)[VIDEO_IDX[0]:VIDEO_IDX[1]]):
        path = ORIGINAL_PATH + value
        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()
        count = 0
        while success:
            if count % TAKE_FRAME == 0:
                image = cv2.resize(image, SIZE)
                cv2.imwrite(POSITIVES_PATH + value.split('.')[0] + f'_frame{count}.jpg', image)
            success,image = vidcap.read()
            count += 1

    # process other videos
    print('Processing class2 videos...')
    for value in tqdm(os.listdir(NORMAL_PATH)[VIDEO_IDX[0]:VIDEO_IDX[1]]):
        path = NORMAL_PATH + value
        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()
        count = 0
        while success:
            if count % TAKE_FRAME == 0:
                image = cv2.resize(image, SIZE)
                cv2.imwrite(NEGATIVES_PATH + value.split('.')[0] + f'_frame{count}.jpg', image)
            success,image = vidcap.read()
            count += 1
    print(f'Final data size estimate: {get_size() * 1e-6} mb')
    
if __name__ == '__main__':
    initialize_dirs();
    # check if the videos actually exist
    os.listdir(ORIGINAL_PATH)[6:8];
    # check if the videos actually exist
    os.listdir(NORMAL_PATH)[6:8]
    
    generate_data('data/Train/Class1/', 'data/Train/Class2/', VIDEO_IDX = TRAIN_SIZE)
    print('=== Finished processing training videos ===')
    generate_data('data/Val/Class1/', 'data/Val/Class2/', VIDEO_IDX = VAL_SIZE)
    print('=== Finished processing validation videos ===')
    #generate_data('data/Test/', 'data/Test/', VIDEO_IDX = TEST_SIZE)
