# GCU_OSS_Team57 (202434734 김성준, 202434735 김성현, 202434754 김현수, 202434851 최서웅)
This project implements a real-time facial expression recognition system using OpenCV, Dlib, and Keras.
It captures video from a webcam to detect faces and analyze facial expressions, classifying them into one of seven categories: 
Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. OpenCV's Haar Cascade Classifier is used for face detection,
Dlib's 68 facial landmarks model extracts key facial features, and a Keras-based deep learning model predicts the facial expression.
To run this project, you need Python 3.6 or higher and the following libraries: OpenCV, Dlib, Keras, and NumPy.
Additionally, three pre-trained files are required:
haarcascade_frontalface_default.xml for face detection, shape_predictor_68_face_landmarks.dat for facial landmark extraction, and emotion_model.hdf5 for expression classification.
When executed, the program activates the webcam and displays detected faces with green rectangles, along with the recognized expression above each face. 
To exit the program, press the Esc key. The system performs better in well-lit environments, and the accuracy depends on the quality of the emotion recognition model.
