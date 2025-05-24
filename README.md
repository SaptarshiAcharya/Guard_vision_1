# 📹 Smart CCTV Surveillance System with YOLOv8

This is a Flask-based intelligent CCTV surveillance system that integrates the **YOLOv8** object detection model to monitor video streams in real-time. The system detects **humans and/or animals**, records footage, and sends **email alerts with intrusion snapshots** upon detecting suspicious activity.

---

## 🚀 Features

- Real-time object detection using **YOLOv8**
- Supports local and IP camera video streams
- Multiple detection modes:
  - Human Detection
  - Animal Detection
  - Both
- Automatic video recording and snapshot capture on detection
- Sends **email alerts** with attached images of intrusions
- Cooldown logic to limit excessive email alerts
- Web-based interface for live monitoring and control
- RESTful API with CORS enabled

---

## 🛠️ Requirements

- Python 3.8+
- Install dependencies with:

pip install -r requirements.txt

yaml
Copy
Edit

Recommended dependencies:

Flask
Flask-Cors
opencv-python
ultralytics

yaml
Copy
Edit



## 🔧 Setup & Usage

### 1. Clone the Repository

git clone 
cd cctv-surveillance-system

shell
Copy
Edit

### 2. Install Dependencies

pip install -r requirements.txt

shell
Copy
Edit

### 3. Run the Application

python app.py

yaml
Copy
Edit

The app will start at `http://localhost:5000`.

### 4. Access the Web Interface

Navigate to: [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🌐 API Endpoints

| Endpoint         | Method | Description                             |
|------------------|--------|-----------------------------------------|
| `/start`         | POST   | Start surveillance with given config     |
| `/stop`          | POST   | Stop surveillance                       |
| `/status`        | GET    | Get current system status               |
| `/video_feed`    | GET    | MJPEG stream of the live video feed     |


🧠 Detection Details
Detection is performed using YOLOv8n (nano) for speed

Only every 3rd frame is processed for object detection

Intrusions are logged when a person, dog, or cat is detected (based on mode)
