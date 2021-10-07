"""
The key purpose of this script is to obtain the video frames for our dataset.
    We are obtaining 30 videos for each exercise, at 30 frames per video. 
    We are able to adjust this number depending on whether or not it is effective 
        enough for our model.

Once the keypoints for each frame are extracted, we save them as numpy arrays 
in the respective video folder. 

"""

# Import libraries
import sys
import cv2
import mediapipe_interpreter
import numpy as np
import folder_setup
from PIL import Image as im


'''
This function creates the dataset by obtaining the video for every exercise and then 
applying the pose model on it. This way, the co-ordinates for every frame are synthesised 
into a numpy array and saved in the dataset. There are 30 co-ordinate arrays for every video 
(1 array per frame) and 30 videos for every exercise. 

    Inputs: results: co-ordinates of joints/landmarks 
    Outputs: 258 attributes of co-ordinates  

'''
def create_data(filename=None):

    if filename is None:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        req_fps = 30/(frame_count)
    count = 0 

    # Declare mediapipe model being used (0.5 for both parameters is ideal) 
    with mediapipe_interpreter.mp_holistic.Holistic(min_detection_confidence=0.5, 
                            min_tracking_confidence=0.5) as holistic:

        # Loop through exercises
        for exercise in folder_setup.exercises:
            # Loop through videos
            for video in range(folder_setup.start_folder, folder_setup.start_folder+folder_setup.no_videos):
                # Loop through video length
                for frame_num in range(folder_setup.sequence_length):

                    # Read from feed
                    success, frame = cam.read()
                    #frame = cv2.resize(frame, [640, 480])
                    #frame = cv2.flip(frame, 1)

                    if success:
                        cv2.imwrite(folder_setup.os.path.join(folder_setup.DATA_PATH, exercise, str(video), '{}.jpg').format(frame_num),frame)
                        if filename is None:
                            count += 30 # i.e. at 30 fps, this advances one second
                        else:
                            count += req_fps
                        cam.set(1, count)

                    # Make detections
                    image, results = mediapipe_interpreter.convert_detections(frame, holistic)

                    # Plot landmarks
                    mediapipe_interpreter.plot_styled_landmarks(image, results)

                    # Check if beginning of recording video
                    if frame_num == 0: 
                        cv2.putText(image, 'COLLECTING NEW VIDEO', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(exercise, video), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 35, 255), 1, cv2.LINE_AA)
                        # Display on screen
                        cv2.imshow('OpenCV Feed', image)
                        # Wait for 2 seconds to get in position
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(exercise, video), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 35, 255), 1, cv2.LINE_AA)
                    
                    # Display on screen
                    cv2.imshow('Webcam+Pose Feed', image)

                    # Export attributes
                    keypoints = mediapipe_interpreter.extract_attributes(results)
                    npy_path = folder_setup.os.path.join(folder_setup.DATA_PATH, exercise, str(video), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully if Q is hit
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cam.release()
        cv2.destroyAllWindows()

    cam.release()
    cv2.destroyAllWindows()


'''
This function displays frames by obtaining the video and then applying the pose model on it. 
This is just for testing purposes, and should run with the webcam if the script is called.

'''
def display_data():
    cam = cv2.VideoCapture(0)
    # Declare mediapipe model being used (0.5 for both parameters is ideal)
    with mediapipe_interpreter.mp_holistic.Holistic(min_detection_confidence=0.5, 
                            min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            # Read from feed
            success, frame = cam.read()

            # Make detections
            image, results = mediapipe_interpreter.convert_detections(frame, holistic)

            # Draw landmarks
            mediapipe_interpreter.plot_styled_landmarks(image, results)

            # Display on screen
            cv2.imshow('Webcam+Pose Feed', image)

            # Break gracefully if Q is hit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_data(folder_setup.os.path.join(folder_setup.os.getcwd(), 'videos','toeraises','tR','25.avi'))