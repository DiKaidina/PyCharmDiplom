# Diploma Work Documentation

The project is dedicated to creating a machine learning model based on a deep neural network using Mediapipe and Tensorflow frameworks.



## Acknowledgements

 - [Mediapipe](https://developers.google.com/mediapipe)
 - [Tensorflow](https://www.tensorflow.org/?hl=ru)
 - [Keras](https://keras.io/)


## To Run the Project Locally

Clone the project

```bash
  git clone https://link-to-project
```

For Python Interpreter configuration use ```venv``` folder as a manually configured virtual environment with pre-installed libraries and packages.

### Packages versions used in the project:
- Mediapipe 0.10.14
- Streamlit 1.34.0
- Keras 3.3.3
- Tensorflow 2.16.1
- Numpy 1.26.0
- OpenCv 1.0.1

For local Python Interpreter install dependencies

```bash
  pip install mediapipe 
  pip install tensorflow
  pip install streamlit
  pip install keras
  pip install tensorflow
  pip install numpy 
  pip install opencv

```

To run the main script ```streamlit_recognition.py``` via streamlit 

```bash
  streamlit run streamlit_recognition.py 
```


## Project structure

The project consists of 3 script files -   ``` streamlit_recognition.py ```,  ```gestures_collection.py```, ```model_fit.py```. Model weights are loaded into ```weights``` folder. The main configuration file of the project is ```venv```.

``` streamlit_recognition.py ``` is a main file that rund the neural network model.  The file consisits of several functions:

```def get_points(param) ``` - collects the key points of the both hands, face and body and saves them into .npy files

```draw_hands(image, results)``` - draws the key points of the hands

```draw_face(image, results)``` - draws the key points of the face

```draw_pose(image, results)``` - draws the key points of the body

```def recognize_gestures()``` - collects images from camera and runs the model for making live predictions



```gestures_collection.py``` is a separated Python script for collecting new gestures if needed.

```def create_directories_from_array(array)``` - creates local directories for gestures based on labels from the input array.

```model_fit.py``` is responsible for training a model based on input labels (gestures)

```def label_map(gestures)```  collects all of the input gestures into one array and gives every single gesture a separate number for convinience

```def config_model(gestures)``` is a confuguration of the neural network with compilation




## Features of the project

- Real-time gesture recognition
- Рand-assembled dataset of 30 kazakh gestures
- LSTM RNN architecture



## Authors

- [@Diana Kaidina](https://github.com/DiKaidina)
