import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, render_template

app = Flask(__name__)

# 加载人脸检测级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载人脸情绪识别模型
model = tf.keras.models.load_model('./emotion_detection_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
input_size = 48

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return "No image file provided", 400
    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected image file", 400
    # 保存上传的图片到临时文件
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)
    # 使用 OpenCV 和模型进行情绪检测
    image = cv2.imread(temp_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_img, (input_size, input_size))
        normalized_face = resized_face / 255.0
        expanded_face = np.expand_dims(normalized_face, axis=-1)
        expanded_face = np.expand_dims(expanded_face, axis=0)
        prediction = model.predict(expanded_face)
        emotion_label = np.argmax(prediction)
        cv2.putText(image, emotion_labels[emotion_label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # 保存标注后的图片为临时文件
    result_image_path = "result_image.jpg"
    cv2.imwrite(result_image_path, image)
    # 删除上传的临时图片
    os.remove(temp_image_path)
    # 返回标注后的图片
    with open(result_image_path, 'rb') as f:
        result_image = f.read()
    os.remove(result_image_path)
    return Response(result_image, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)