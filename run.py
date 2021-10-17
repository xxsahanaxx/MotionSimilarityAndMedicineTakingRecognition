# Ninger's imports
from medicine_taking.Run import VGGRNN

# Sahana’s imports
from exercise_recognition import train as exercise_train
from exercise_recognition import test

import torch
import sys, os, warnings
import argparse

warnings.filterwarnings('ignore')
 
CAM_CONSTANT = 0
 
def run_combined(context):
    # If the model path does not exist, create it
    if not os.path.exists(os.path.join(context.model_path)):
        os.mkdir(os.path.join(context.model_path))
 
    # If the model name is not entered, throw an error
    if not context.model_name:
        print("Need a model name to store weights to, please input it using --model_name and try again!")
        sys.exit()

    # CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    if context.model_type == "VGG":
        CAM_CONSTANT = 0
        FROM_WEBCAM = True # run the webcam
        VGGRNN()
        
    if context.model_type != "VGG": # the LSTM model/GRU model
        # Change variables
        exercise_train.RNN_type = arguments.model_type
        exercise_train.loc = os.getcwd()

        # Define the model
        model = exercise_train.define_model(exercise_train.RNN_type)
        print(model.summary())

        # Load the model weights
        model = exercise_train.load_model(model, arguments.model_name, arguments.model_path)
        
        # Pass it through to test script 
        test.test_detections(model)
 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_type', type=str, choices=['VGG','LSTM','GRU'])
 
    arguments = parser.parse_args()
    run_combined(arguments)
