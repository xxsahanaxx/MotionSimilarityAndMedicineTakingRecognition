"""
The biggest drawback of mediapipe is that there isn't much documentation on the algorithm, even though 
the website gives a lot of insight into a simple example. 

Hence, the key purpose of this script is to decode 'mediapipe' and its built-in functions and try to 
dissect them in a way that is understandable to the user.
    These include creating helper functions that can assist in a method in a more straight-forward manner. 

"""

# Import libraries
import mediapipe as mp
import cv2
import numpy as np

# Using the holistic model of mediapipe and drawing tools 
mp_holistic = mp.solutions.holistic 
mp_drawings = mp.solutions.drawing_utils 

'''
This function reads in a frame of video being captured, and makes detections 
using the pose estimation model being fed into it. 

    Inputs: image: frame being captured
            model: being estimated with 
    Outputs: image: being fed in
             results: co-ordinates of joints 
'''
# Make all detections on the frame of video being captured 
def convert_detections(image, model):
    # OpenCV uses BGR, and model requires RGB, so we convert 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Make image unwritable while feeding into model to save changes and prevent alteration of data
    image.flags.writeable = False    
    # Apply the model prediction on the image               
    results = model.process(image) 
    # Change writability of image so that it can be displayed afterwards                
    image.flags.writeable = True       
    # Convert back to BGR for OpenCV to use             
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

'''
This function uses results obtained from pose estimation, and draws landmarks on
the image frame. 

    Inputs: image: frame being captured
            results: co-ordinates of joints/landmarks 
    Outputs: NA - draws on image
'''
def plot_styled_landmarks(image, results):
    # Plot pose landmarks in a specific colour, thickness and point radius 
    mp_drawings.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawings.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawings.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Plot left hand landmarks in a specific colour, thickness and point radius 
    mp_drawings.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawings.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawings.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Plot right hand landmarks in a specific colour, thickness and point radius  
    mp_drawings.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawings.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawings.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

'''
This function uses results obtained from pose estimation to extract x, y, and z dimensional 
co-ordinates. It then congregates them as an array to store as a dataset entity.  

    Inputs: results: co-ordinates of joints/landmarks 
    Outputs: 258 attributes of co-ordinates  
'''
def extract_attributes(results):
    # Obtain the x, y, z and visibility co-ordinates for the main skeleton if they exist, else 0
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Left hand and right hand x, y, z co-ordinates
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # Return these attributes 
    return np.concatenate([pose, lh, rh])

