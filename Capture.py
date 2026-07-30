#!/usr/bin/env python3
"""
MAIN FACE CAPTURE - Captures only the main (closest) face at 0.1s intervals
"""

import cv2
import time
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
RESIDENT_DIR = BASE_DIR / "train" / "resident_1"
RESIDENT_DIR.mkdir(parents=True, exist_ok=True)

# Find the correct haarcascade path
haar_paths = [
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    '/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
    '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
]

face_cascade = None
for path in haar_paths:
    if os.path.exists(path):
        face_cascade = cv2.CascadeClassifier(path)
        print(f"✅ Loaded face detector from: {path}")
        break

if face_cascade is None or face_cascade.empty():
    print("❌ Haarcascade not found! Please install opencv-data")
    print("   Run: sudo apt-get install opencv-data")
    exit()

print("="*60)
print("📸 MAIN FACE CAPTURE - 0.1s INTERVALS")
print("="*60)
print(f"  Saving to: {RESIDENT_DIR}")
print("  Captures only the MAIN (closest) face")
print("  Press ESC to quit")
print("="*60 + "\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

# Find next available number
count = 0
existing = list(RESIDENT_DIR.glob("face_*.jpg"))
if existing:
    numbers = [int(f.stem.replace('face_', '')) for f in existing if f.stem.replace('face_', '').isdigit()]
    count = max(numbers) + 1 if numbers else 0

print(f"✅ Camera ready! Starting from photo #{count}")
print("📸 Stand in front of camera - capturing main face only!\n")

capture_interval = 0.1
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    # Find the main face (closest/biggest)
    main_face = None
    main_area = 0
    
    for (x, y, w, h) in faces:
        area = w * h
        if area > main_area:
            main_area = area
            main_face = (x, y, w, h)
    
    # Draw only the main face
    if main_face is not None:
        x, y, w, h = main_face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "MAIN FACE", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Crop main face for saving
        face_roi = frame[y:y+h, x:x+w]
        
        # Auto-capture at 0.1s intervals
        current_time = time.time()
        if (current_time - last_capture_time) >= capture_interval:
            filename = RESIDENT_DIR / f"face_{count}.jpg"
            cv2.imwrite(str(filename), face_roi)
            print(f"✅ Captured: face_{count}.jpg (main face)")
            count += 1
            last_capture_time = current_time
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 10)
            cv2.imshow("Main Face Capture", frame)
            cv2.waitKey(20)
    
    # Show count and info
    cv2.putText(frame, f"Photos: {count}  |  ESC=Quit", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if main_face is None:
        cv2.putText(frame, "No face detected", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Capturing every 0.1s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Main Face Capture", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📸 Captured {count} photos of the main face!")
print(f"   Saved to: {RESIDENT_DIR}")
print("\nNext: Run python3 Train.py to retrain the model!")
