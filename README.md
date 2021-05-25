### Auto encoder for image reconstruction
This repository contains code for image reconstruction using a ConvNet auto encoder model. The model in auto_H.py takes an image and reconstrut to generate the same input image eventhough the image is different from the type of the training data. For example, the model was not trained with people images but when an input of an iimage of a person is given, the model generates it. 

#### Requirements

- Python 3.7
- Tensorflow 2.0
- CUDA Version: 11.1

#### Training
- The training data in ./trainH/ are taken from [WHU-RS19](https://www.kaggle.com/sunray2333/whurs191) dataset.
- In order to train the model from scratch, set the "global_option" variable in auto_H.py file to "trainH" (global_option = "trainH") then run the auto_H.py file.

#### Testing
- The pretrained model is provided ./weights_H.hdf5
- To test the pretrained model, set the "global_option" variable in auto_H.py file to "testH" (global_option = "testH") then run the auto_H.py file.
- Result:



