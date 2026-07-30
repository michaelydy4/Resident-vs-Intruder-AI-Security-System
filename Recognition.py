#!/usr/bin/env python3
"""
PRODUCTION-GRADE TRACKING - Kalman Filter + IoU Matching
"""

import cv2
import pickle
import time
import face_recognition
from pathlib import Path
from collections import deque
from datetime import datetime
from jetson_inference import detectNet
from jetson_utils import videoSource, cudaToNumpy
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "face_model.pkl"
CAPTURE_DIR = BASE_DIR / "Captured_Images"

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Higher threshold = more likely to recognize you!
MATCH_THRESHOLD = 0.30  # Increased from 0.30 to 0.45
STABILIZATION_SECONDS = 3
MEMORY_SECONDS = 5.0
IOU_THRESHOLD = 0.3


class FaceTracker:
    def __init__(self, box_id, x1, y1, x2, y2):
        self.id = box_id
        self.history = deque(maxlen=30)
        self.label = "Unknown"
        self.locked = False
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.captured = False
        self.current_prediction = "Unknown"
        self.counted = False
        
        self.last_x1, self.last_y1 = x1, y1
        self.last_x2, self.last_y2 = x2, y2
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2
        self.width = x2 - x1
        self.height = y2 - y1
        
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.statePre = np.array([[self.center_x], [self.center_y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[self.center_x], [self.center_y], [0], [0]], np.float32)
        self.has_prediction = False
    
    def predict_position(self):
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0]
    
    def update_position(self, x1, y1, x2, y2):
        self.last_x1, self.last_y1 = x1, y1
        self.last_x2, self.last_y2 = x2, y2
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2
        self.width = x2 - x1
        self.height = y2 - y1
        self.last_seen = time.time()
        
        measurement = np.array([[self.center_x], [self.center_y]], np.float32)
        self.kalman.correct(measurement)
        self.has_prediction = True
    
    def get_box(self):
        if self.has_prediction:
            pred_x, pred_y = self.predict_position()
            return int(pred_x - self.width/2), int(pred_y - self.height/2), int(pred_x + self.width/2), int(pred_y + self.height/2)
        return self.last_x1, self.last_y1, self.last_x2, self.last_y2
    
    def add_prediction(self, label):
        if label not in ["No Face", "Error", "Unknown"]:
            self.history.append(label)
            self.current_prediction = label
    
    def get_majority_vote(self):
        if not self.history:
            return "Unknown"
        from collections import Counter
        return Counter(self.history).most_common(1)[0][0]
    
    def is_expired(self):
        return time.time() - self.last_seen > MEMORY_SECONDS
    
    def get_age(self):
        return time.time() - self.first_seen


class TrackerManager:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.stats = {"resident": 0, "intruder": 0}
        self.counted_ids = set()
    
    def compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou
    
    def update(self, detections, frame, label_func):
        current_boxes = []
        
        for detection in detections:
            x1 = int(detection.Left)
            y1 = int(detection.Top)
            x2 = int(detection.Right)
            y2 = int(detection.Bottom)
            box = (x1, y1, x2, y2)
            current_boxes.append(box)
            
            matched_id = None
            best_iou = IOU_THRESHOLD
            
            for tid, tracker in self.trackers.items():
                pred_box = tracker.get_box()
                iou = self.compute_iou(box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_id = tid
            
            if matched_id is not None:
                tracker = self.trackers[matched_id]
                tracker.update_position(x1, y1, x2, y2)
                
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size > 0:
                    label = label_func(person_roi)
                    tracker.add_prediction(label)
                
                if not tracker.locked and tracker.get_age() >= STABILIZATION_SECONDS:
                    tracker.label = tracker.get_majority_vote()
                    tracker.locked = True
                    
                    if not tracker.counted:
                        tracker.counted = True
                        self.counted_ids.add(matched_id)
                        if tracker.label == "Resident":
                            self.stats["resident"] += 1
                        elif tracker.label == "Intruder":
                            self.stats["intruder"] += 1
                    
                    if tracker.label == "Intruder" and not tracker.captured:
                        self.save_intruder_face(frame, x1, y1, x2, y2)
                        tracker.captured = True
                
                pred_x1, pred_y1, pred_x2, pred_y2 = tracker.get_box()
                yield (pred_x1, pred_y1, pred_x2, pred_y2, tracker.label, tracker.locked, tracker.current_prediction, tracker.id)
            
            else:
                tracker_id = str(self.next_id)
                self.next_id += 1
                tracker = FaceTracker(tracker_id, x1, y1, x2, y2)
                self.trackers[tracker_id] = tracker
                
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size > 0:
                    label = label_func(person_roi)
                    tracker.add_prediction(label)
                
                yield (x1, y1, x2, y2, "New", False, tracker.current_prediction, tracker_id)
        
        for tid in list(self.trackers.keys()):
            if self.trackers[tid].is_expired():
                if tid in self.counted_ids:
                    self.counted_ids.remove(tid)
                del self.trackers[tid]
    
    def save_intruder_face(self, frame, x1, y1, x2, y2):
        try:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return
            face_crop = cv2.resize(face_crop, (200, 200))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = CAPTURE_DIR / f"intruder_{timestamp}.jpg"
            cv2.imwrite(str(filename), face_crop)
            print(f"📸 INTRUDER CAPTURED: {filename.name}")
        except Exception as e:
            pass
    
    def get_stats(self):
        return self.stats


class FaceRecognizer:
    def __init__(self, model_path):
        self.known_encodings = []
        self.load_model(model_path)
    
    def load_model(self, model_path):
        if not model_path.exists():
            print("❌ Model not found!")
            return False
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'encodings' in data:
            self.known_encodings = data['encodings']
            print(f"✅ Loaded {len(self.known_encodings)} known faces")
            return True
        return False
    
    def classify_face(self, face_roi):
        if not self.known_encodings:
            return "No Model"
        
        try:
            small_roi = cv2.resize(face_roi, (0, 0), fx=0.2, fy=0.2)
            rgb_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_roi)
            face_encodings = face_recognition.face_encodings(rgb_roi, face_locations)
            
            if not face_encodings:
                return "No Face"
            
            for encoding in face_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                if len(distances) > 0:
                    min_distance = min(distances)
                    if min_distance <= MATCH_THRESHOLD:
                        return "Resident"
                    else:
                        return "Intruder"
            
            return "Intruder"
        except:
            return "Error"


def main():
    recognizer = FaceRecognizer(MODEL_PATH)
    if not recognizer.known_encodings:
        return
    
    print("\n" + "="*60)
    print("🎯 PRODUCTION-GRADE TRACKING")
    print("="*60)
    print(f"  Match Threshold: {MATCH_THRESHOLD} (higher = easier to recognize)")
    print(f"  Stabilization: {STABILIZATION_SECONDS}s")
    print(f"  Memory: {MEMORY_SECONDS}s")
    print(f"  IoU Threshold: {IOU_THRESHOLD}")
    print("  Press 'q' to quit")
    print("="*60 + "\n")
    
    net = detectNet(model="ssd-mobilenet-v2", threshold=0.5)
    camera = videoSource("/dev/video0")
    
    tracker_manager = TrackerManager()
    
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: 0.0"
    
    while True:
        img = camera.Capture()
        if img is None:
            continue
        
        frame = cudaToNumpy(img)
        frame_count += 1
        
        detections = net.Detect(img)
        
        for x1, y1, x2, y2, label, locked, pred, tid in tracker_manager.update(detections, frame, recognizer.classify_face):
            if locked:
                if label == "Resident":
                    color = (0, 255, 0)
                elif label == "Intruder":
                    color = (0, 0, 255)
                else:
                    color = (200, 200, 200)
            else:
                if pred == "Resident":
                    color = (0, 255, 100)
                elif pred == "Intruder":
                    color = (0, 100, 255)
                else:
                    color = (255, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if locked:
                display_label = f"🔒 {label}"
            else:
                display_label = f"⏳ {pred}"
            
            cv2.putText(frame, display_label, (x1, max(y1 - 10, 0)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = 30 / elapsed if elapsed > 0 else 0
            fps_display = f"FPS: {fps:.1f}"
            start_time = time.time()
        
        stats = tracker_manager.get_stats()
        status = f"Resident: {stats['resident']} | Intruder: {stats['intruder']} | Trackers: {len(tracker_manager.trackers)}"
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, fps_display, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "q:quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"\nResidents: {stats['resident']} | Intruders: {stats['intruder']}")

if __name__ == "__main__":
    main()
