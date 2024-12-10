# facial expression recognition
import cv2
import dlib
import numpy as np
from keras.models import load_model

# face recognition
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Returns the positions of eyes, nose, mouth, etc. for facial expression recognition
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# facial expression labeling
expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# facial expression weight model
model = load_model('./emotion_model.hdf5')

# run video
video_capture = cv2.VideoCapture(0)

prev_faces = []

while True:
    # return ret, frame 
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Gray conversion for face recognition video start
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face recognition
    # The closer the scaleFactor is to 1, the better facial expression recognition is. The farther away the scaleFactor is, the worse it is.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # When a face is recognized within a region, facial expressions are recognized.
    for (x, y, w, h) in faces:
        # Draw a square to fit the size of your face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # return face size
        face_roi = gray[y:y+h, x:x+w]

        # Convert the same size as the facial expression dataset to recognize facial expressions
        # An error occurs if the size of the dataset image and the input face are different.
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # Analyzing facial expressions through models
        output = model.predict(face_roi)[0]

        # Return the value of the corresponding expression
        expression_index = np.argmax(output)

        # Save label value according to facial expression
        expression_label = expression_labels[expression_index]
        
        # Expression value output
        cv2.putText(frame, expression_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  # output
    cv2.imshow('Expression Recognition', frame)

    # Exit by pressing esc
    key = cv2.waitKey(25)
    if key == 27:
        break

if video_capture.Opened():
    video_capture.release()
cv2.destroyAllWindows()
