"""
This script serves the purpose of testing RNN models for both LSTM and GRU models, 
and allows the user to either test them from a test data subset or in real-time using
the webcam. 
    The test data subset evaluation evaluates model based on 4 metrics: confusion
    matrix, accuracy, precision, and F1

When this script is run from terminal, by default, it calls both LSTM and GRU
pre-trained models. 

"""
import mediapipe_interpreter 
import train
import folder_setup
import preprocess
import csv
import cv2
import os
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, f1_score

loc = os.getcwd()
colors = [(199,117,96), (190,202,113), (140,224,99), (129,197,227), (79,152,248)]

'''
The purpose of this function is to create a probability wizard that can generate
the degree of accurate prediction and draw them on the image. 
'''
def probability_wizard(actions, input_frame, colors, results1, results2=None):
    output_frame = input_frame.copy()
    for num, prob in enumerate(results1):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*300), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    if results2 is not None: 
        for num, prob in enumerate(results2):
            cv2.rectangle(output_frame, (539,60+num*40), (int(539+prob*300), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (539, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

'''
This function tests predictions on the 'test' subset of data. It then produces 
a confusion matrix, accuracy score, precision score and F1 score to help compare 
model performances using metrics.
    The per sample predictions and true results are stored inside a CSV file 
    called 'testPred<LSTM/GRU>.csv'.

'''
def test_predictions_on_test_set(loc, model, RNN_type):

    # Obtain train and test data from split 
    X_train, X_test, y_train, y_test = preprocess.load_train_data(loc)
    print("The training and testing sets are of sizes: ", X_train.shape, X_test.shape)

    # Print the structure of the model 
    print(model.summary())

    # Making test predictions on X_test
    yhat = model.predict(X_test)

    # Encode the predictions by categorical label instead of one-hot-encoding (0,1,2,3,4)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    file_path = folder_setup.os.path.join(folder_setup.os.getcwd(), "res", "testPred"+RNN_type+".csv")
    # Save predictions made on test samples in a CSV
    with open(file_path, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ytrue','yhat'])
        for i, j in zip(ytrue,yhat):
            writer.writerow([i,j])

    # Display testing metrics 
    print("Confusion Matrix: \n", multilabel_confusion_matrix(ytrue, yhat))
    print("Accuracy: {}%".format(accuracy_score(ytrue, yhat)*100))
    print("Precision: {}%".format(precision_score(ytrue, yhat, average='macro')*100))
    print("F1: {}%".format(f1_score(ytrue, yhat, average='macro')*100))


'''
This function tests detections using the webcam in real time. It collates the 
predictions of one (by default) or two models and compares them beside each other.
    It uses the Mediapipe Holistic pose model to obtain landmarks and their 258
    co-ordinates. These are then extracted and stored in a 'sequence' containing
    30 frames (as of now uncustomisable). 
    
A sequence constantly re-initialises itself to only hold the last 30 frames of 
    landmark co-ordinates. 
    
'''
def test_detections(model1, model2=None):
    # Create detection variables
    sequence = []
    sentence1 = []
    sentence2 = []
    predictions1 = []
    predictions2 = []
    # Probability wizard kicks in when this threshold is crossed by any prediction
    threshold = 0.5

    cam = cv2.VideoCapture(0)
    print("Live feed has been set up")

    # Set mediapipe model 
    with mediapipe_interpreter.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():

            # Read from camera feed
            success, frame = cam.read()

            # Make detections on frame
            image, results = mediapipe_interpreter.convert_detections(frame, holistic)
            
            # Draw landmarks
            mediapipe_interpreter.plot_styled_landmarks(image, results)
            
            # Extract co-ordinates and append to the sequence
            keypoints = mediapipe_interpreter.extract_attributes(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            # If full sequence is obtained
            if len(sequence) == 30:
                # Pass through one video at a time
                res1 = model1.predict(np.expand_dims(sequence, axis=0))[0]
                print("Model1= ", res1)
                # Append results to prediction list
                predictions1.append(np.argmax(res1))

                # If another model is passed for comparison, again append 
                # to another prediction list
                if(model2 is not None):
                    res2 = model2.predict(np.expand_dims(sequence, axis=0))[0]
                    print("Model2= ", res2)
                    predictions2.append(np.argmax(res2))
            
                # Check if the last 8 predictions are different from current
                if np.unique(predictions1[-8:])[0]==np.argmax(res1): 
                    # If result surpasses threshold, check if current continuous predictions have more 
                    # than 1 prediction
                    if res1[np.argmax(res1)] > threshold: 
                        # If sentence has more than one prediction, check if prev prediction is not the same as current
                        if len(sentence1) > 0: 
                            if folder_setup.exercises[np.argmax(res1)] != sentence1[-1]:
                                sentence1.append(folder_setup.exercises[np.argmax(res1)])
                        else:
                            sentence1.append(folder_setup.exercises[np.argmax(res1)])
                    
                    # Same for model2
                    if model2 is not None:
                        if res2[np.argmax(res2)] > threshold: 
                            if len(sentence2) > 0: 
                                if folder_setup.exercises[np.argmax(res2)] != sentence2[-1]:
                                    sentence2.append(folder_setup.exercises[np.argmax(res2)])
                            else:
                                sentence2.append(folder_setup.exercises[np.argmax(res2)])

                # Only store last 3 predictions of sequence
                if len(sentence1) > 3: 
                    sentence1 = sentence1[-3:]
                
                if len(sentence2) > 3: 
                    sentence2 = sentence2[-3:]

                # Display probability wizard windows on screen
                image = probability_wizard(folder_setup.exercises, image, colors, res1, res2)
                
            # Put attributes on frame to display to GUI
            cv2.rectangle(image, (0,0), (320, 40), (178, 163, 221), -1)
            cv2.putText(image, "LSTM", (0,35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, ''.join(sentence1[-1:]), (100,35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if model2 is not None:
                cv2.rectangle(image, (640,0), (1020, 40), (178, 163, 221), -1)
                cv2.putText(image, "GRU", (640,35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, ''.join(sentence2[-1:]), (740,35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Show to screen
            cv2.imshow('Test Camera Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model1 = train.define_model("LSTM")
    model1 = train.load_model(model1,"LSTMexercises.h5",os.path.join(loc,"models"))
    #test_predictions_on_test_set(loc,model1)
    model2 = train.define_model("GRU")
    model2 = train.load_model(model2,"GRUexercises.h5",os.path.join(loc,"models"))
    #test_predictions_on_test_set(loc,model2)

    test_detections(model1,model2)