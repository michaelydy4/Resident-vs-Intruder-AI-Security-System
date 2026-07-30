Resident vs Intruder AI Security System



Overview:
This project is a real-time face recognition security system built for the NVIDIA Jetson Nano.
"Resident" or an unrecognized "intruder." A key achievement of this project was significantly improving 
performance through DECTNet. The system now operates at 30 FPS with 99.3% accuracy, compared to the 
CPU-only approach, which achieved only 3 FPS and 80% accuracy. When an intruder is detected, the system 
automatically captures and downloads their image to the Captured_Images/ directory. Another feature 
Is that it? It includes a 3-second stabilization period where, when it detects a face, it sometimes flickers. 
between resident and intruder, so the system will get the majority vote of whether the target is a resident 
or an intruder in these 3 seconds and lock the label to stop the label from flickering. It also has Kalman Filter-based 
tracking for smooth face tracking and position memory to maintain tracking and label even when individuals 
Briefly move out of frame.


Core Features:
1. Real-time Detection, 30 fps using DECTNet
2. Face Recognition, 99.3% accuracy with ResNet-18
3. Intruder Face Capture: saves the face photo of an intruder
4. Smooth Tracking, Kalman Filter for Face Following
5. Multiple Residents: Support any number of residents.
6. Train Auto: save/Train stop and resume, auto-saves every 50 cycles, and can stop and resume training every time.

How it works (Recognition.py):
1. The camera captures the video from the camera.
2. DECTNet finds people.
3. Face-Extracted Form of Each Person
4. Recognition compares to known residents. 
5. After 3 seconds it will lock the label for that person.

Train process:
1. Capture any amount of photos of the resident's face (resident)
2. Takes 2000 random faces from 6000 random faces (intruder)
3. Transfer learning with ResNet-18
4. Training cycles for the amount you need 
5. Achieve 99+ percent accuracy

Technology info:
Hardware: NVIDIA Jetson Nano 4GB, USB Webcam, 64GB SD Card
Detection: DECTNet 
Recognition: ResNet-18
Optimization: TensorRT, CUDA 
Tracking: Kalman Filter
Language: Python 3.10

How to use:
1. Capture
- Run "python3 Capture.py"
- Stand in front of the camera
- The system will automatically capture your face every 0.1 seconds
- Photos are saved to "train/resident_1/"
- Press ESC when done

Tip: Take at least 100 photos from different angles for best result

2. Train
- Run "python3 Train.py" to start training
- It will first ask you for the cycles, the default is 2400 rounds(press enter for default or enter a number you want)
- Enter "infinite" for infinite trains and control C to end
- Training will show accuracy after every 10 cycles
- Press Control C to stop and save progress
- The best model is saved as "face_model_best.pkl" (For now its the modle for my face)

3. Recognition
- Run "python3 Recognition.py"
- A window will open showing the camera feed
- Resident -> Green box, "Resident"
- Intruder -> Red box, "Intruder"
- Intruder faces are automatically saved to "Captured_Images/"

Controls:
- Press "q" to quit
- Press "s" to save the current frame


Future Improvements:
- Web interface
- Email alerts
- Named residents

Acknowledgments
- NVIDIA DLI
- ID Tech
- My Instructors(Blaze, Flash)

