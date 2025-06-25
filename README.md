# 👁️ Eye Gaze Tracking - Speech Type Classifier

This project classifies spoken communication as either **"Scripted"** or **"Natural"** using eye-tracking data extracted from videos. The classification is performed by analyzing blinking behavior (via Eye Aspect Ratio), gaze direction, and head movement.

🎓 **Project for:** Machine Learning II  
🏛️ **University:** Université Paris-Est Créteil (UPEC)  
👨‍🏫 **Supervisor:** Prof. Amine Nait-Ali  
👩‍💻👨‍💻 **Authors:** Yazan Hasan & Modan Mohan Sarker  
📅 **Date:** January 2025  

---

## 🚀 Features

- Upload video and analyze speech behavior using:
  - Eye blink detection (EAR)
  - Gaze direction tracking
  - Head position analysis
- Classify speech as **scripted** or **natural** using a DNN model
- Real-time GUI built with **PyQt5**
- Graphical plots of gaze and head movement
- Results displayed in an intuitive interface

---

## 🧠 Model Overview

- Framework: **TensorFlow**
- Model Type: **Deep Neural Network**
- Input: Preprocessed video frames (64×64 RGB)
- Output: Binary classification → Scripted (0) / Natural (1)
- Saved model file: `speech_type_classifier.h5`

---

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- dlib (for facial landmarks)
- PyQt5
- Matplotlib

---

## 🖥️ How to Run

1. Clone the repository
2. Install dependencies via `requirements.txt`
3. Ensure you have the following files:
   - `shape_predictor_68_face_landmarks.dat`
   - `speech_type_classifier.h5`
4. Run the GUI:
   ```bash
   python main.py
