import smtplib
from email.message import EmailMessage
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime

# Load the TensorFlow model
model_dir = r"C:\Users\user\Downloads\centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\saved_model"
model = tf.saved_model.load(model_dir)


def detect_objects(image_np, model):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections


def send_email_alert():
    msg = EmailMessage()
    msg.set_content(
        'Intrusion detected! Please check the surveillance system.')
    msg['Subject'] = 'Intrusion Alert!'
    msg['From'] = 'tempemail.123@outlook.com'
    msg['To'] = 'tempemail.123@outlook.com'
    server = smtplib.SMTP('smtp.office365.com', 587)
    server.starttls()  # Upgrade the connection to TLS
    server.login('tempemail.123@outlook.com', 'your_password')
    server.send_message(msg)
    server.quit()


def is_intrusion_time():
    current_hour = datetime.now().hour
    return current_hour >= 22 or current_hour < 7


def real_time_detection(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_objects(frame, model)
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        if any((classes == 1) & (scores > 0.5)) and is_intrusion_time():
            send_email_alert()
        if classes[i] == 1 and scores[i] > 0.5:
            box = boxes[i] * [frame.shape[0], frame.shape[1],
                              frame.shape[0], frame.shape[1]]
            cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(
                box[3]), int(box[2])), (0, 255, 0), 2)
        cv2.imshow('Real-time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_detection(model)
