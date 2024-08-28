import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template

app = Flask(__name__)

# 加载人脸检测级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载人脸情绪识别模型
model = tf.keras.models.load_model('./emotion_detection_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
input_size = 48


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face_img, (input_size, input_size))
            normalized_face = resized_face / 255.0
            expanded_face = np.expand_dims(normalized_face, axis=-1)
            expanded_face = np.expand_dims(expanded_face, axis=0)
            prediction = model.predict(expanded_face)
            emotion_label = np.argmax(prediction)
            cv2.putText(frame, emotion_labels[emotion_label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)