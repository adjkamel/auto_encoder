### Auto encoder for image reconstruction
This repository contains code for image reconstruction using a ConvNet auto encoder model. The model in auto_H.py takes an image and reconstracts it and generate the same input image even though the image is different from the type of the training data. For example, the model was not trained with people images but when an input of an iimage of a person is given, the model generates it. 

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

![image](https://user-images.githubusercontent.com/50513215/119574838-fdcd1580-bdad-11eb-90ba-b00b8a1cf1cf.png)

![image](https://user-images.githubusercontent.com/50513215/119574963-2fde7780-bdae-11eb-8db8-2e9152c2cfc6.png)

