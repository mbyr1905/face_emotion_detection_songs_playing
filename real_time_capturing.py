import cv2
import time
from keras.models import model_from_json
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import os
import random

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

emotion_detected = None  
start_time = None  
temp_label=None
flag=0

songs_path ="D:\\face_emotion_detection\\songs\\songs"

def switch_case(label_detected):
    audio_path = os.path.join(songs_path, label_detected)
    random_number = str(random.randint(1, 4))
    if label_detected == "sad":
        print("Sad")
        audio_file_path=audio_path+f"\\{random_number}.mp3"
        # print(audio_file_path)
        os.startfile(audio_file_path)
    elif label_detected == "happy":
        print("happy")
        audio_file_path=audio_path+f"\\{random_number}.mp3"
        # print(audio_file_path)
        os.startfile(audio_file_path)
    elif label_detected == "neutral":
        print("neutral")
        audio_file_path=audio_path+f"\\{random_number}.mp3"
        print(audio_file_path)
        os.startfile(audio_file_path)
    elif label_detected == "angry":
        print("angry")
        audio_file_path=audio_path+f"\\{random_number}.mp3"
        # print(audio_file_path)
        os.startfile(audio_file_path)
    else:
        print("This is the default case")

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Display emotion on the image
            cv2.putText(im, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            
            # Capture image after 8 seconds and detect emotion
            if emotion_detected is None:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > 8:
                    temp_label = prediction_label
                    webcam.release()
                    cv2.destroyAllWindows()
                    flag=1
                    break
            else:
                cv2.imshow("Captured Image", im)
        if flag==1:
            break

    except cv2.error:
        pass
    if flag==1:
        break
    cv2.imshow("Output", im)
    
    key = cv2.waitKey(1)
    if key == 27:
        break


webcam.release()
cv2.destroyAllWindows()


switch_case(temp_label) 