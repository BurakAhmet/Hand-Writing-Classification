# Handwriting Digit Recognition
This project enables the recognition of handwritten digits using TensorFlow and Tkinter libraries with mnist dataset. After drawing a digit on the Tkinter interface, the TensorFlow model is used to predict the drawn digit.

I used **CNNs** (Convolutional Neural Networks) and **data augmentation** techniques to get high val-accuracy result.

## Preview
https://github.com/BurakAhmet/Hand-Writing-Digit-Recognition/assets/89780902/bb9a05ea-e746-45fa-a76c-be1dc44fb1ad

## Model
I used CNN (Convolutional Neural Networks) and data augmentation techniques in my model
### Model Accuracy
Final training loss: **0.0444**

Final training accuracy: **0.9858**

Final validation loss: **0.0182** 

Final validation accuracy: **0.9948**

![model accuracy](https://github.com/BurakAhmet/Hand-Writing-Digit-Recognition/assets/89780902/c2566e6c-ea26-4f98-b929-b43317bc8828)

## Technologies Used
* Python 3: The project is developed using Python programming language.
* Pillow (PIL): Utilized for capturing and processing images.
* TensorFlow: Used for training the data, loading pre-trained models and making predictions.
* NumPy: Employed for array manipulation and normalization of input data.
* Tkinter: Utilized for creating the user interface (canvas).
* Google Colab: Used for fast model training with GPUs.
