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

    Inputs: filename: video file path that is used to obtain co-ordinates from
                if no argument is passed, the webcam is used to record the videos  

'''
def create_data(filename=None):

    if filename is None:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        req_fps = ((frame_count)/45)
    count = 0 

    # Declare mediapipe model being used (0.5 for both parameters is ideal) 
    with mediapipe_interpreter.mp_holistic.Holistic(min_detection_confidence=0.5, 
                            min_tracking_confidence=0.5) as holistic:

        # Loop through exercises
        for exercise in folder_setup.exercises:
            # Loop through videos
            for video in range(folder_setup.start_folder, folder_setup.no_videos+1):
                # Loop through video length
                for frame_num in range(folder_setup.sequence_length):

                    # Read from feed
                    success, frame = cam.read()
                    #frame = cv2.resize(frame, [640, 480])
                    #frame = cv2.flip(frame, 1)

                    if success:
                        cv2.imwrite(folder_setup.os.path.join(folder_setup.DATA_PATH, exercise, str(video), '{}.jpg').format(frame_num),frame)

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

                    if filename is None:
                        count += 30 # i.e. at 30 fps, this advances one second
                    else:
                        count += req_fps
                    cam.set(1, count)

                    # Break gracefully if Q is hit
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cam.release()
        cv2.destroyAllWindows()

    cam.release()
    cv2.destroyAllWindows()


'''
This function opens another pop-up window using OpenCV and displays frames by obtaining 
the video and then applying the pose model on it. 
    This is just for testing purposes, and should run with the webcam if the script is called.

    Inputs: filetype: "image" or "video"
                if webcam is used, leave this parameter empty
            filename: video file path that is used to display co-ordinate map on
                if webcam is used, leave this parameter empty

'''
def display_data(filetype=None, filename=None):

    if filename is not None and filetype != "image":
        cam = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
    else: 
        cam = cv2.VideoCapture(0)

    # Declare mediapipe model being used (0.5 for both parameters is ideal)
    with mediapipe_interpreter.mp_holistic.Holistic(min_detection_confidence=0.5, 
                            min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            # Read from feed
            success, frame = cam.read()

            if(filename is not None and filetype == "image"):
                frame = cv2.imread(filename)

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
    display_data()