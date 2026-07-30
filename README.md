# Resident vs. Intruder AI Security System

A real-time face recognition security system built for the NVIDIA Jetson Nano that detects and identifies people as either a recognized "Resident" or an unrecognized "Intruder" at 30 FPS with 99.3% accuracy.

![Resident vs Intruder Demo](https://img.shields.io/badge/Demo-Video_Link_Coming_Soon-red)

## The Algorithm

This system uses a combination of computer vision and deep learning to perform real-time face recognition:

1. **Person Detection**: The system uses **DetectNet (SSD-Mobilenet-v2)** with TensorRT acceleration to detect people in the camera feed at 30 FPS. This runs on the Jetson's GPU for maximum speed. Using DetectNet significantly improved performance from 3 FPS (CPU-only) to 30 FPS.

2. **Face Extraction**: Once a person is detected, their face is cropped from the frame and resized for recognition.

3. **Face Recognition**: Using **ResNet-18** with transfer learning, the system compares the detected face against known resident faces. Each face is converted to a 128-dimensional encoding vector, and the distance between vectors determines if there's a match. The system achieves 99.3% accuracy.

4. **3-Second Stabilization**: A majority voting system prevents label flickering. The system observes the face for 3 seconds, takes a majority vote of the predictions, and locks the label to stop flickering between "Resident" and "Intruder."

5. **Kalman Filter Tracking**: Predicts face position for smooth tracking, even when individuals turn their heads or move around.

6. **Position Memory**: Maintains the tracking label even when individuals briefly move out of frame.

7. **Intruder Capture**: When an intruder is detected, their face is automatically saved to the `Captured_Images/` directory for security review.

### Training Process
- Captured 500+ photos of the resident's face from different angles and lighting conditions
- Used 2000+ random faces from a pool of 6000 as intruders
- Transfer learning with ResNet-18
- Auto-saves every 50 training cycles
- Can stop and resume training at any time
- Achieved 99.3% accuracy after training

## Screenshots

### Resident Detection
![Resident](https://raw.githubusercontent.com/michaelydy4/Resident-vs-Intruder-AI-Security-System/main/images/Screenshot%202026-07-30%20093524.png)

### Intruder Detection
![Intruder](https://raw.githubusercontent.com/michaelydy4/Resident-vs-Intruder-AI-Security-System/main/images/Screenshot%202026-07-30%20100053.png)

### Training Results
![Training](https://raw.githubusercontent.com/michaelydy4/Resident-vs-Intruder-AI-Security-System/main/images/Screenshot%202026-07-30%20093110.png)

### Captured Intruder Images
![Captured](https://raw.githubusercontent.com/michaelydy4/Resident-vs-Intruder-AI-Security-System/main/images/Screenshot%202026-07-30%20100106.png)

### Dependencies
- Python 3.10+
- face_recognition
- scikit-learn
- OpenCV
- PyTorch
- TensorRT
- CUDA
- jetson-inference

## Running This Project

### Prerequisites
- NVIDIA Jetson Nano (JetPack installed)
- USB Webcam
- Python 3.10+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/michaelydy4/Resident-vs-Intruder-AI-Security-System.git
cd Resident-vs-Intruder-AI-Security-System
