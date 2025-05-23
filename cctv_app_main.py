from flask import Flask, render_template, request, Response, jsonify
import cv2
import datetime
import os
import threading
import smtplib
import imghdr
from email.message import EmailMessage
from ultralytics import YOLO
from flask_cors import CORS
import logging
import time

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories for saving footage and intrusions
FOOTAGE_DIR = "FOOTAGE"
INTRUSION_DIR = "INTRUSIONS"
os.makedirs(FOOTAGE_DIR, exist_ok=True)
os.makedirs(INTRUSION_DIR, exist_ok=True)

# Define email cooldown (in seconds)
EMAIL_COOLDOWN = 120  # 2 minutes

class CCTVSystem:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.video_url = ""
        self.recipient_email = ""
        self.mode = "Both"
        self.running = False
        self.cap = None
        self.out = None
        self.last_email_sent = None
        self.last_image_saved = None
        self.frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.detection_interval = 3
        self.email_count = 0
        self.email_lock = threading.Lock()  # Lock dedicated to email logic

    def start_surveillance(self, video_url, recipient_email, mode):
        if self.running:
            return False

        # Build the URL if provided; otherwise, use a local camera (0)
        self.video_url = f"http://{video_url}:8080/video" if video_url else 0
        self.recipient_email = recipient_email
        self.mode = mode
        self.running = True
        self.frame = None  # Reset previous frame

        self.cap = cv2.VideoCapture(self.video_url)
        if not self.cap.isOpened():
            self.running = False
            logger.error("Failed to open video stream.")
            return False

        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        video_filename = os.path.join(FOOTAGE_DIR, f'{current_date}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(video_filename, fourcc, 10,
                                   (int(self.cap.get(3)), int(self.cap.get(4))))

        # Start frame processing in a separate thread.
        threading.Thread(target=self.process_frames, daemon=True).start()
        return True

    def stop_surveillance(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        self.cap = None
        self.out = None
        self.frame = None
        # Reset the email timer and counter when surveillance is stopped.
        self.last_email_sent = None
        self.email_count = 0

    def process_frames(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                logger.error("Video capture is not available.")
                break

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from capture.")
                break

            processed_frame = frame.copy()
            intrusion_detected = False
            detected_labels = set()
            self.frame_count += 1

            # Run detection only on every Nth frame if mode is not "Normal CCTV".
            if self.mode != "Normal CCTV" and (self.frame_count % self.detection_interval == 0):
                try:
                    resized = cv2.resize(frame, (640, 480))
                    results = self.model(resized)
                    ratio_x = frame.shape[1] / 640
                    ratio_y = frame.shape[0] / 480

                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes:
                            for box in result.boxes.data.tolist():
                                if len(box) < 6:
                                    continue
                                x1, y1, x2, y2 = map(int, [box[0] * ratio_x, box[1] * ratio_y,
                                                             box[2] * ratio_x, box[3] * ratio_y])
                                cls_id = int(box[5])
                                label = self.model.names.get(cls_id, "unknown")

                                if self._check_detection_criteria(label):
                                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    detected_labels.add(label)
                                    intrusion_detected = True
                except Exception as e:
                    logger.error(f"Inference error: {e}")

            if intrusion_detected:
                # Run heavy tasks in a separate thread: wait 5 sec, capture full-body image, save, email.
                threading.Thread(target=self._handle_intrusion,
                                 args=(processed_frame.copy(), detected_labels.copy()),
                                 daemon=True).start()

            if self.out:
                self.out.write(processed_frame)

            with self.lock:
                self.frame = processed_frame.copy()

    def _check_detection_criteria(self, label):
        if self.mode == "Human Detection":
            return label == "person"
        elif self.mode == "Animal Detection":
            return label in ["dog", "cat"]
        elif self.mode == "Both":
            return label in ["person", "dog", "cat"]
        return False

    def _handle_intrusion(self, frame, detected_labels):
        # Wait for 5 seconds before capturing the frame for a full-body image.
        time.sleep(5)
        # Capture a live frame after the delay; if unavailable, fall back to the initial frame.
        full_body_frame = self.get_frame()
        if full_body_frame is None:
            full_body_frame = frame

        current_time = datetime.datetime.now()

        # Save the captured full-body frame if enough time has elapsed.
        if self.last_image_saved is None or (current_time - self.last_image_saved).total_seconds() > 120:
            for i in range(3):
                timestamp = current_time.strftime('%Y-%m-%d_%H-%M-%S')
                path = os.path.join(INTRUSION_DIR, f'intrusion_{timestamp}_{i}.jpg')
                cv2.imwrite(path, full_body_frame)
                logger.info(f"Image saved: {path}")
            self.last_image_saved = current_time

        # Use the dedicated email lock to prevent concurrent email sending.
        with self.email_lock:
            # Reset email counter if cooldown has passed.
            if self.last_email_sent is None or (current_time - self.last_email_sent).total_seconds() > EMAIL_COOLDOWN:
                self.email_count = 0

            # Only send email if fewer than 3 emails have been sent in the current cycle.
            if self.email_count < 3:
                self._send_email(", ".join(detected_labels))
                self.last_email_sent = current_time
                self.email_count += 1
            else:
                logger.info("Email limit reached; waiting for cooldown.")

    def _send_email(self, intrusion_type):
        sender_email = "YOUR_EMAIL"
        sender_password = "YOUR_PASSWORD"

        msg = EmailMessage()
        msg.set_content(f"Intrusion detected: {intrusion_type}")
        msg['Subject'] = "CCTV Intrusion Alert"
        msg['From'] = sender_email
        msg['To'] = self.recipient_email

        images = sorted(
            [os.path.join(INTRUSION_DIR, f) for f in os.listdir(INTRUSION_DIR) if f.endswith(".jpg")],
            reverse=True)[:3]

        for img_path in images:
            with open(img_path, 'rb') as f:
                img_data = f.read()
                img_type = imghdr.what(img_path)
                msg.add_attachment(img_data, maintype='image', subtype=img_type,
                                   filename=os.path.basename(img_path))

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            logger.info("Email sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# Initialize the CCTV system
cctv_system = CCTVSystem()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/start', methods=['POST'])
def start():
    data = request.get_json()
    success = cctv_system.start_surveillance(
        video_url=data.get('video_url', ''),
        recipient_email=data.get('recipient_email', ''),
        mode=data.get('mode', 'Both')
    )
    return jsonify({'success': success})

@app.route('/stop', methods=['POST'])
def stop():
    cctv_system.stop_surveillance()
    return jsonify({'success': True})

@app.route('/status')
def status():
    return jsonify({
        'running': cctv_system.running,
        'mode': cctv_system.mode,
        'video_url': cctv_system.video_url,
        'recipient_email': cctv_system.recipient_email
    })

def generate_frames():
    # Continuously yield frames for streaming.
    while True:
        if cctv_system.running:
            frame = cctv_system.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
