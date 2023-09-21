import numpy as np
import tensorflow as tf
import cv2

# Load the TensorFlow model
model_dir = r"C:\Users\user\Downloads\centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\saved_model"
model = tf.saved_model.load(model_dir)


def detect_objects(image_np, model):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections


def filter_frames_with_persons(video_path, model, output_path="output.avi"):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_with_persons = []
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
        if any((classes == 1) & (scores > 0.5)):
            frames_with_persons.append(frame)
    video.release()
    if frames_with_persons:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames_with_persons:
            out.write(frame)
        out.release()


if __name__ == "__main__":
    video_path = r"C:\Users\user\Downloads\8387402966_202309150025330000000.mp4"
    output_path = "filtered_video.mp4"
    filter_frames_with_persons(video_path, model, output_path)
    print("done")
