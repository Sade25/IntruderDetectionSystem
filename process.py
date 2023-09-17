import numpy as np
import tensorflow as tf
import cv2
print(tf.__version__)
# Load the TensorFlow model
# model_dir = r"C:\Users\user\Downloads\centernet_hg104_512x512_coco17_tpu-8\saved_model"
model_dir = r"C:\Users\user\Downloads\efficientdet_d7_coco17_tpu-32\saved_model"
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


if __name__ == "__main__":
    video_path = r"C:\Users\user\Downloads\8387402966_202309150025330000000.mp4"
    # video_path = r"C:\Users\user\Downloads\5084643772_202309141528560000000.mp4"
    output_path = "filtered_video.mp4"
    filter_frames_with_persons(video_path, model, output_path)
    print("done")
