import sys
import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QWidget)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from tensorflow.keras.models import load_model


class FrameProcessor(QThread):
    processed_frame = pyqtSignal(np.ndarray)
    processed_data = pyqtSignal(float, float, float, float, str)
    processing_done = pyqtSignal()  # Signal to indicate processing is done
    result_updated = pyqtSignal(str)  # Signal to emit the running result

    def __init__(self, cap, predictor, detector, model, EAR_THRESHOLD=0.25, calibration_frames=10, process_frequency=12):
        super().__init__()
        self.cap = cap
        self.predictor = predictor
        self.detector = detector
        self.model = model
        self.EAR_THRESHOLD = EAR_THRESHOLD
        self.process_frequency = process_frequency
        self.running = True
        self.paused = False
        self.calibration_frames = calibration_frames
        self.calibrated = False
        self.blink_threshold_dynamic = EAR_THRESHOLD

        # Analysis variables
        self.frame_count = 0
        self.gaze_x, self.gaze_y = 0, 0
        self.head_x, self.head_y = 0, 0
        self.scripted_score = 0

        # Gaze direction analysis
        self.center_gaze_duration = 0
        self.off_center_gaze_duration = 0

        # Calibration variables for EAR threshold
        self.calibration_ears = []

        # Store predictions
        self.predictions = []

    def run(self):
        while self.running and self.cap.isOpened():
            if self.paused:
                self.msleep(100)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            # Process every nth frame
            if self.frame_count % self.process_frequency == 0:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    if not self.calibrated and len(self.calibration_ears) < self.calibration_frames:
                        self.calibrate_blink_threshold(landmarks)
                    else:
                        if not self.calibrated:
                            self.blink_threshold_dynamic = np.mean(self.calibration_ears) - 0.05
                            self.calibrated = True
                        self.track_gaze_direction(landmarks)
                        self.track_head_position(landmarks)

                # Determine speech type based on gaze direction
                speech_type = self.predict_speech_type(frame)
                self.predictions.append(speech_type)

                # Calculate the percentage of "Scripted" and "Natural" predictions
                total_frames = len(self.predictions)
                scripted_count = self.predictions.count("Scripted")
                natural_count = self.predictions.count("Natural")
                scripted_percentage = (scripted_count / total_frames) * 100
                natural_percentage = (natural_count / total_frames) * 100

                # Emit the running result
                result_text = f"Speech Type Detected: {speech_type}"
                self.result_updated.emit(result_text)

                # Emit the processed frame and data back to the main GUI thread
                self.processed_frame.emit(frame)
                self.processed_data.emit(self.gaze_x, self.gaze_y, self.head_x, self.head_y, speech_type)

            self.frame_count += 1

        # Emit processing done signal
        self.processing_done.emit()

    def calibrate_blink_threshold(self, landmarks):
        LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]
        ear = self.compute_ear(landmarks, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS)
        self.calibration_ears.append(ear)

    def compute_ear(self, landmarks, left_eye_points, right_eye_points):
        def eye_aspect_ratio(eye_points):
            A = np.linalg.norm(np.array([landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y]) -
                               np.array([landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y]))
            B = np.linalg.norm(np.array([landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y]) -
                               np.array([landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y]))
            C = np.linalg.norm(np.array([landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y]) -
                               np.array([landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y]))
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        return (left_ear + right_ear) / 2.0

    def track_gaze_direction(self, landmarks):
        left_eye_center = landmarks.part(36)
        right_eye_center = landmarks.part(45)

        eye_center_x = (left_eye_center.x + right_eye_center.x) / 2
        eye_center_y = (left_eye_center.y + right_eye_center.y) / 2

        self.gaze_x, self.gaze_y = eye_center_x, eye_center_y

        img_w, img_h = self.cap.get(cv.CAP_PROP_FRAME_WIDTH), self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        center_x_min, center_x_max = img_w * 0.4, img_w * 0.6
        center_y_min, center_y_max = img_h * 0.4, img_h * 0.6

        if center_x_min < eye_center_x < center_x_max and center_y_min < eye_center_y < center_y_max:
            self.center_gaze_duration += 1
        else:
            self.off_center_gaze_duration += 1

    def track_head_position(self, landmarks):
        nose_point = landmarks.part(33)  # Nose tip for head position
        img_w, img_h = self.cap.get(cv.CAP_PROP_FRAME_WIDTH), self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.head_x = nose_point.x / img_w
        self.head_y = nose_point.y / img_h

    def predict_speech_type(self, frame):
        # Resize the frame to (64, 64) as expected by the model
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (64, 64))

        # Normalize the frame (values between 0 and 1)
        frame = frame / 255.0

        # Expand dimensions to match the model's input shape (1, 30, 64, 64, 3)
        frame = np.expand_dims(frame, axis=0)
        frame = np.expand_dims(frame, axis=0)

        # Predict
        prediction = self.model.predict(frame)

        # Speech type based on prediction
        if prediction[0][0] > 0.5:
            return "Scripted"
        else:
            return "Natural"

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class EyeGazeTrackingApp(QMainWindow):
    def __init__(self):
        super(EyeGazeTrackingApp, self).__init__()

        self.setWindowTitle("Eye Gaze Tracking Application")
        self.setGeometry(100, 100, 1400, 900)

        # Set up the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)

        # Title Banner
        title_label = QLabel("Eye Gaze Tracking Application")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #00468b; padding: 15px;")
        main_layout.addWidget(title_label)

        # Video Display and Status Group Box
        video_status_group = QGroupBox("Video and Status Information")
        video_status_layout = QHBoxLayout()

        # Video Display
        self.videoDisplayLabel = QLabel()
        self.videoDisplayLabel.setFixedSize(800, 450)
        self.videoDisplayLabel.setStyleSheet("border: 3px solid #00468b; border-radius: 15px; background-color: #dbe7f3;")

        # Status Information
        self.status_group = QGroupBox("Status Information")
        self.status_layout = QVBoxLayout()
        self.statusLabel = QLabel("Status: Ready")
        self.speechTypeLabel = QLabel("Speech Type: ")

        self.statusLabel.setFont(QFont("Arial", 16))
        self.speechTypeLabel.setFont(QFont("Arial", 16, QFont.Bold))
        self.statusLabel.setStyleSheet("padding: 10px; color: #333;")
        self.speechTypeLabel.setStyleSheet("padding: 10px; color: #555;")

        self.status_layout.addWidget(self.statusLabel)
        self.status_layout.addWidget(self.speechTypeLabel)
        self.status_group.setLayout(self.status_layout)

        # Combine Video Display and Status Information
        video_status_layout.addWidget(self.videoDisplayLabel)
        video_status_layout.addWidget(self.status_group)
        video_status_group.setLayout(video_status_layout)

        # Control Buttons Group Box
        self.control_group = QGroupBox("Controls")
        self.control_layout = QGridLayout()

        self.uploadButton = QPushButton("Upload Video")
        self.startAnalysisButton = QPushButton("Start Analysis")
        self.pauseButton = QPushButton("Pause")
        self.resumeButton = QPushButton("Resume")
        self.resetButton = QPushButton("Reset")
        self.plotButton = QPushButton("Plot Movement")

        # Add buttons to the grid layout
        self.control_layout.addWidget(self.uploadButton, 0, 0)
        self.control_layout.addWidget(self.startAnalysisButton, 0, 1)
        self.control_layout.addWidget(self.pauseButton, 1, 0)
        self.control_layout.addWidget(self.resumeButton, 1, 1)
        self.control_layout.addWidget(self.resetButton, 2, 0)
        self.control_layout.addWidget(self.plotButton, 2, 1)

        self.control_group.setLayout(self.control_layout)

        # Add everything to the main layout
        main_layout.addWidget(video_status_group)
        main_layout.addWidget(self.control_group)

        # Set styles for buttons and labels
        self.setStyleSheet("""
            QPushButton {
                background-color: #00468b;
                color: white;
                padding: 15px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                min-width: 150px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
            QLabel {
                font-size: 16px;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 18px;
                color: #333;
                margin-top: 25px;
                border: 3px solid #00468b;
                border-radius: 10px;
                padding: 15px;
            }
        """)

        # Set up face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Load the machine learning model
        self.model = load_model("speech_type_classifier.h5")

        # Variables to store gaze and head movement data
        self.gaze_positions_x = []
        self.gaze_positions_y = []
        self.head_positions_x = []
        self.head_positions_y = []

        self.cap = None

        # Connect buttons to functions
        self.uploadButton.clicked.connect(self.load_video)
        self.startAnalysisButton.clicked.connect(self.start_analysis)
        self.pauseButton.clicked.connect(self.pause_analysis)
        self.resumeButton.clicked.connect(self.resume_analysis)
        self.resetButton.clicked.connect(self.reset_analysis)
        self.plotButton.clicked.connect(self.plot_movement)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.cap = cv.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.statusLabel.setText("Status: Error opening video file.")
                return

            self.statusLabel.setText("Status: Video Loaded")
            self.startAnalysisButton.setEnabled(True)

    def start_analysis(self):
        if self.cap is None or not self.cap.isOpened():
            self.statusLabel.setText("Status: Load a video first.")
            return

        self.statusLabel.setText("Status: Analyzing...")

        # Start the frame processor in a separate thread
        self.processor = FrameProcessor(self.cap, self.predictor, self.detector, self.model)
        self.processor.processed_frame.connect(self.display_frame)
        self.processor.processed_data.connect(self.update_data)
        self.processor.processing_done.connect(self.analysis_finished)
        self.processor.result_updated.connect(self.update_result)  # Connect the new signal
        self.processor.start()

    def display_frame(self, frame):
        qt_image = self.convert_cv_qt(frame)
        self.videoDisplayLabel.setPixmap(qt_image)

    def update_data(self, gaze_x, gaze_y, head_x, head_y, speech_type):
        self.speechTypeLabel.setText(f"Speech Type: {speech_type}")

        # Store gaze and head positions for plotting
        self.gaze_positions_x.append(gaze_x)
        self.gaze_positions_y.append(gaze_y)
        self.head_positions_x.append(head_x)
        self.head_positions_y.append(head_y)

    def update_result(self, result_text):
        # Update the status label with the running result
        self.statusLabel.setText(f"Status: Analyzing... {result_text}")

    def analysis_finished(self):
        self.statusLabel.setText("Status: Analysis Complete")
        self.plotButton.setEnabled(True)  # Enable the plot button after analysis

    def pause_analysis(self):
        if hasattr(self, 'processor') and self.processor.isRunning():
            self.processor.pause()
            self.statusLabel.setText("Status: Paused")

    def resume_analysis(self):
        if hasattr(self, 'processor') and self.processor.isRunning():
            self.processor.resume()
            self.statusLabel.setText("Status: Resumed")

    def reset_analysis(self):
        if hasattr(self, 'processor') and self.processor.isRunning():
            self.processor.stop()
        self.gaze_positions_x = []
        self.gaze_positions_y = []
        self.head_positions_x = []
        self.head_positions_y = []
        self.statusLabel.setText("Status: Ready")
        self.speechTypeLabel.setText("Speech Type: ")
        self.plotButton.setEnabled(False)  # Disable the plot button after reset

    def plot_movement(self):
        if not self.gaze_positions_x:
            self.statusLabel.setText("No data to plot. Run the analysis first.")
            return

        fig, axs = plt.subplots(4, 1, figsize=(10, 12))

        # Plot gaze positions
        axs[0].plot(self.gaze_positions_x, label="Gaze X Position", color="#00468b")
        axs[0].set_title("Gaze X Position Over Time", fontsize=16)
        axs[0].legend()

        axs[1].plot(self.gaze_positions_y, label="Gaze Y Position", color="#005bb5")
        axs[1].set_title("Gaze Y Position Over Time", fontsize=16)
        axs[1].legend()

        # Plot head positions
        axs[2].plot(self.head_positions_x, label="Head X Position", color="#0073e6")
        axs[2].set_title("Head X Position Over Time", fontsize=16)
        axs[2].legend()

        axs[3].plot(self.head_positions_y, label="Head Y Position", color="#3399ff")
        axs[3].set_title("Head Y Position Over Time", fontsize=16)
        axs[3].legend()

        plt.tight_layout()
        plt.show()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_image = qt_image.scaled(self.videoDisplayLabel.width(), self.videoDisplayLabel.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(qt_image)

    def closeEvent(self, event):
        if hasattr(self, 'processor'):
            self.processor.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = EyeGazeTrackingApp()
    main_window.show()
    sys.exit(app.exec_())