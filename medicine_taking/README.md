# Video-Classification-in-Real-Time

Video classification using VGG16 as a feature extractor and seasoning with RNN. 

Dataset used is self-created (including water drinking and taking medicine).

> A simple LSTM model is used to better classify temporal frame sequences from videos.

Drinking water          |  Taking Medicine
:-------------------------:|:-------------------------:
![drinking1](https://user-images.githubusercontent.com/61758760/137617994-a5770f41-c982-475c-a927-70e1b0fd534d.gif) | ![TakingMedicine1](https://user-images.githubusercontent.com/61758760/137618000-5c29585b-eca6-49c5-930f-7bda6652dba1.gif)



--- 

> Some of the use cases would be monitoring anomalies, suspicious human actions, alerting the staff/authorities.

## Table of Contents
* [Background Theory](#background-theory)
* [Running Inference](#running-inference)
* [Pipeline](#pipeline)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
* [References](#references)
* [Next Steps](#next-steps)

## Background Theory
**Feature extraction:**
- After fine tuning/unfreezing its four top layers, VGG16 is employed as a feature extractor.
- After that, a basic movement classifier is coupled to VGG16 and trained to determine if the frame belongs to Drinking water or Taking Medicine. 
- The top classifier is then disconnected, and the sparse representations of each frame are obtained using only a dense layer with a 1024 output size.


- **Data to lstm format:** For each video frame, the sparse representations are stacked into a tensor of size (NUM_FRAMES, LOOK_BACK, 1024). 

<div align="center">
<img src="https://github.com/Ninger-Gong/MotionSimilarityAndMedicineTakingRecognition/blob/main/medicine_taking/VGG16_Structure.png" width=570>
<p>- VGG 16 Model architecture -</p>
</div>

---

**RNN:**
- A standard LSTM is used. 
- The model is using LSTM in the build_model(), Note that you need GPU/CUDA support if you would like to run CUDnnLSTM layers in the model. 
- Finally, the LSTM network is trained to distinguish between your desired water drinking and taking medicine videos.

## Running Inference
- Install all the required Python dependencies:
```
pip install -r requirements.txt
```
- To run inference either on a test video file or on webcam: 
```
python run.py 
```
- Note that the inference is set on the test video file by default. 
- To change it, simply set ``` FROM_WEBCAM = True ``` in the config. options at mylib/Config.py
- Trained model weights (for this example) can be downloaded from from the folder called weights. Make sure you extract them into the folder 'weights'.
- The class probabilities and inference time per frames is also displayed:
```
[INFO] Frame acc. predictions: 0.6396240741014482
Frame inference in 0.0770 seconds
```

<div align="center">
<img src="https://github.com/Ninger-Gong/MotionSimilarityAndMedicineTakingRecognition/blob/main/medicine_taking/LSTM_Structure.png" width=500>
<p>- LSTM model structure -</p>
</div>

The result of testing would be:
Drinking water          |  Taking Medicine
:-------------------------:|:-------------------------:
![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/61758760/137619414-2b9c5c07-00dd-40ac-b4ed-33803c63b5cb.gif) | ![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/61758760/137619423-f40b4ab0-251a-4a26-8793-edc21087a712.gif)

- In case of severe false positivies, make sure to optimize the threshold and positive_frames parameters to further narrow down the predictions. Please refer config.

The threshold of prediction accuracy can be changed, based on the actual testing result
```
Threshold = 0.60
if pred >= Threshold:
```

## Pipeline

### Preprocessing:
- Some image processing is required before training on your own data videos
- In 'Preprocessing.ipynb' file, the frames from each video classes are extracted and sorted into respective folders.
- but please run the Preprocessing.py to deal with your own data on your own laptop
- Note that the frames are resized to 224x224 dimensions (in order to fit with VGG16 input layer size).
- The dataset can be download from here: https://drive.google.com/drive/folders/1p2It1lzbfLQafYkBmRklXd7Oa3l5HQ0w
- with the image_data set

### Training:
- 'Train.ipynb', as the name implies trains your model.
- Training is visualized with the help of TensorBoard. Use the command:
```
%reload_ext tensorboard
%tensorboard --logdir='/content/drive/Shareddrives/Part IV Research Project/Medicine Action/data/_training_logs/rnn' --host=127.0.0.1
```
<div align="center">
<img src="https://github.com/Ninger-Gong/MotionSimilarityAndMedicineTakingRecognition/blob/main/medicine_taking/Training_Result.png" width=470>
<p>- Training accuracy -</p>
</div>

### Please notice the folder of this part should like:
- medicine_taking
  - weights
  - data
     - Train
     - Val
     - _training_logs
  - dataset
     - Class1
     - Class2
  - tests
  - mylib
  - Train.py
  - Train.ipynb
  - Run.py
  - Run.ipynb
  - requirements.txt
  - Preprocessing.py
  - Preprocessing.ipynb

The folder data will be created after running the Preprocessing.py.
The dataset video should be saved in dataset folder, under two classes folder

- Make sure to check the parameters in config. options at mylib/Config.py
- You will come across the parameters in Train.ipynb, they must be same during the training and inference.
- If you would like to change them, simply do so in the training file and also in config. options.

## References

***Main:***
- VGG16 paper: https://arxiv.org/pdf/1409.1556.pdf
- LSTM paper: https://doi.org/10.1016/j.neunet.2005.06.042

***Optional:***
- TensorBoard: https://www.tensorflow.org/tensorboard

## Next steps
- Change the VGG16 to Inception v3 model
- add more videos for a better performance of the model

<p>&nbsp;</p>

---

## Thanks for the read & have fun!

    - 👯 Clone this repo:
    ```
    $ git clone https://github.com/Ninger-Gong/MotionSimilarityAndMedicineTakingRecognition/blob/main/medicine_taking
    ```

- **Just Run it!**

---

### Reference
This code is based on the code from Sai MJ's repo: https://github.com/saimj7/Action-Recognition-in-Real-Time
