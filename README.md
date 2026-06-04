# Helmetless Bike Rider and Wrong Lane Vehicle Detection System

## Overview

This project is an AI-powered traffic monitoring system designed to detect helmetless bike riders and vehicles moving in the wrong lane in real time. The system utilizes YOLOv8 object detection, object tracking, OCR-based number plate recognition, and a Flask-based dashboard for monitoring and reporting traffic violations.

## Features

* Real-time vehicle detection using YOLOv8
* Helmetless bike rider detection
* Wrong lane vehicle detection
* Vehicle tracking with unique IDs
* License plate recognition using OCR
* Violation image snapshots
* Violation database management using SQLite
* Web dashboard for monitoring and reporting
* Automatic report generation

## System Architecture

1. Video Input
2. YOLOv8 Vehicle Detection
3. Object Tracking
4. Helmet Detection
5. Wrong Lane Detection
6. Number Plate OCR
7. Violation Storage
8. Dashboard Visualization

## Technologies Used

* Python
* YOLOv8
* OpenCV
* Flask
* SQLite
* EasyOCR
* NumPy
* Pandas

## Project Structure

```text
project/
│
├── app.py
├── pipeline.py
├── database.py
├── report.py
├── dashboard.html
├── requirements.txt
├── models/
├── output/
└── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/AbdulMusawair777/Helmetless-bike-rider-and-wrong-lane-vahicle-detection.git
cd Helmetless-bike-rider-and-wrong-lane-vahicle-detection
```

### Create Virtual Environment

```bash
python -m venv env
```

### Activate Environment

Windows:

```bash
env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Results

The system can:

* Detect motorcycles and riders
* Identify riders without helmets
* Detect wrong lane movement
* Extract vehicle registration numbers
* Save violation records in SQLite database
* Generate violation reports

## Future Enhancements

* DeepSORT / ByteTrack Integration
* Multi-camera Support
* Real-time Alerts
* Cloud Deployment
* Traffic Analytics Dashboard

## Author

**Abdul Musawair**

Computer Vision Engineer | AI & Machine Learning Enthusiast

## License

This project is licensed under the Apache 2.0 License.

