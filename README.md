# Real-Time Intrusion Detection System

## Overview
This project is a real-time intrusion detection system that utilizes advanced machine learning models to identify the presence of persons in video feeds. It's designed to work with both prerecorded videos and live camera streams. When a person is detected during the specified intrusion hours (10 pm to 7 am), the system sends an email notification as an alert.

<!--- just ![Demo Image](demo_image.jpg)--->

## Features

- **Real-Time Person Detection**: Uses a TensorFlow-based model to detect persons in video frames with high accuracy.
- **Time-Based Alerts**: Sends intrusion notifications only during specified hours to reduce false alarms.
- **Video Processing**: Can process both live streams and prerecorded videos. Outputs a video containing only the frames with detected persons.
- **Email Notifications**: Automatically sends email alerts when an intrusion is detected.

## How to Use

### Setup

1. Clone the repository: 
git clone https://github.com/your_username/your_repo_name.git


2. Navigate to the project directory:
cd your_repo_name

3. Install the required packages:
pip install -r requirements.txt

4. Set up your email configurations for notifications.

### Running

- For real-time detection:
python real_time_detection.py

- For processing a prerecorded video:
python process_video.py

## Future Improvements

- Integration with home automation systems.
- Support for multi-camera setups.
- Audio analysis to complement video detections.
- Cloud integrations for remote monitoring and control.

## Contributions

Feel free to fork this repository and submit pull requests for any enhancements. All contributions are welcome!

