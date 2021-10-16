"""
This file helps augment data by scaling the dataset to 0.75 and 1.25.

"""
import folder_setup
import numpy as np
import os
import cv2

seventy_five_percent_scale_start = folder_setup.no_videos
hundred_twenty_five_percent_scale_start = folder_setup.no_videos*2

'''
The purpose of this function is to load every frame from every video of every exercise, 
scale it to 0.75 and 1.25 and save both the scaled coordinates and the image in a new folder
either starting from <folder_setup.no_videos*2> or <folder_setup.no_videos*3>. 
    Inputs: scale1 = 0.75 (by default)
            scale2 = 1.25 (by default)

'''
def scale_dataset(scale1=0.75, scale2=1.25):
    # For every action in dataset
    for action in folder_setup.exercises:
        # For every video in action
        for sequence in range(1,folder_setup.no_videos+1):
            try: 
                os.makedirs(os.path.join(folder_setup.DATA_PATH, action, str(sequence+seventy_five_percent_scale_start)), exist_ok=True)
            except:
                pass

            for frame_num in range(folder_setup.sequence_length):
                # Load a frame from original dataset as numpy
                res = np.load(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                # Scale co-ordinates by 0.75
                new_res = res * scale1
                img = cv2.imread(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence), "{}.jpg".format(frame_num)))
                img = cv2.resize(img, (int(img.shape[1] * scale1),int(img.shape[0] * scale1)), interpolation=cv2.INTER_AREA) 
                
                # Save as a new frame in same action
                np.save(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence+seventy_five_percent_scale_start), str(frame_num)), new_res)
                cv2.imwrite(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence+seventy_five_percent_scale_start),str(frame_num)+".jpg"),img)
        
        print("Scaled coordinates for exercise {} by 0.75".format(action))

    # For every action in dataset
    for action in folder_setup.exercises:
        # For every video in action
        for sequence in range(1,folder_setup.no_videos+1):
            try: 
                os.makedirs(os.path.join(folder_setup.DATA_PATH, action, str(sequence+hundred_twenty_five_percent_scale_start)), exist_ok=True)
            except:
                pass

            for frame_num in range(folder_setup.sequence_length):
                # Load a frame from original dataset as numpy
                res = np.load(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                # Scale co-ordinates by 1.25
                new_res = res * scale2
                img = cv2.imread(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence), "{}.jpg".format(frame_num)))
                img = cv2.resize(img, (int(img.shape[1] * scale2),int(img.shape[0] * scale2)), interpolation=cv2.INTER_AREA) 
                
                # Save as a new frame in same action
                np.save(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence+hundred_twenty_five_percent_scale_start), str(frame_num)), new_res)
                cv2.imwrite(folder_setup.os.path.join(folder_setup.DATA_PATH, action, str(sequence+hundred_twenty_five_percent_scale_start),str(frame_num)+".jpg"),img)
        
        print("Scaled coordinates for exercise {} by 1.25".format(action))


if __name__ == "__main__":
    scale_dataset()