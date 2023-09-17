import smtplib
from email.message import EmailMessage
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime


# Load the TensorFlow model
model_dir = r"C:\Users\user\Downloads\centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\saved_model"
# model_dir = r"C:\Users\user\Downloads\centernet_hg104_512x512_coco17_tpu-8\saved_model"
# model_dir = r"C:\Users\user\Downloads\efficientdet_d7_coco17_tpu-32\saved_model"
model = tf.saved_model.load(model_dir)

# This function processes the image and returns detected objects


def detect_objects(image_np, model):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    return detections

# Define a function to filter frames with detected persons


def filter_frames_with_persons(video_path, model, output_path="output.avi"):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    frames_with_persons = []  # List to store frames where a person is detected
    frame_count = 0

    while video.isOpened():
        ret, frame = video.read()
        frame_count += 1
        print(f"Processing frame {frame_count} of {total_frames}...")
        if not ret:
            break

        detections = detect_objects(frame, model)
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()

        # If a person is detected with a confidence above 0.5
        # Adjust the confidence threshold as needed
        if any((classes == 1) & (scores > 0.5)):
            frames_with_persons.append(frame)

    video.release()
    # If there are any frames with detected persons, create the output video
    if frames_with_persons:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames_with_persons:
            out.write(frame)
        out.release()


def real_time_detection(model):
    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        detections = detect_objects(frame, model)
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()

        # Draw bounding boxes on the frame (optional)
        for i in range(len(boxes)):
            # If the object is a person and confidence > 0.5
            # If a person is detected and it's an intrusion based on your criteria
            if any((classes == 1) & (scores > 0.5)) and is_intrusion_time():
                send_email_alert()

            if classes[i] == 1 and scores[i] > 0.5:
                box = boxes[i] * [frame.shape[0], frame.shape[1],
                                  frame.shape[0], frame.shape[1]]
                cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(
                    box[3]), int(box[2])), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-time Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def send_email_alert():
    msg = EmailMessage()
    msg.set_content(
        'Intrusion detected! Please check the surveillance system.')
    msg['Subject'] = 'Intrusion Alert!'
    msg['From'] = 'ahmed.856@osu.edu'
    msg['To'] = 'ahmed.856@osu.edu'

    # Establish a connection to the Outlook SMTP server
    server = smtplib.SMTP('smtp.office365.com', 587)
    server.starttls()  # Upgrade the connection to TLS
    server.login('ahmed.856@osu.edu', 'your_password')
    server.send_message(msg)
    server.quit()


def is_intrusion_time():
    current_hour = datetime.now().hour
    return current_hour >= 22 or current_hour < 7


if __name__ == "__main__":
    # video_path = r"C:\Users\user\Downloads\8387402966_202309150025330000000.mp4"
    # # video_path = r"C:\Users\user\Downloads\5084643772_202309141528560000000.mp4"
    # output_path = "filtered_video.mp4"
    # filter_frames_with_persons(video_path, model, output_path)
    # print("done")
    real_time_detection(model)
