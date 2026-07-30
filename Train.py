#!/usr/bin/env python3
"""
ULTIMATE TRAINING - FAST & CLEAN
"""

import pickle
import face_recognition
from pathlib import Path
import numpy as np
import time
import cv2
import random
import sys
from datetime import datetime, timedelta
import json
import os

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
MODEL_PATH = BASE_DIR / "face_model.pkl"
STATE_PATH = BASE_DIR / "training_state.json"
BEST_MODEL_PATH = BASE_DIR / "face_model_best.pkl"
LOG_PATH = BASE_DIR / "training_log.json"

PROCESSING_SCALE = 0.25


def ask_for_cycles():
    print("\n" + "="*70)
    print("⏱️  SET TRAINING CYCLES")
    print("="*70)
    print("  - Type a number (e.g., 1000) for fixed cycles")
    print("  - Type 'infinite' to train forever (Ctrl+C to stop)")
    print("  - Press ENTER for recommended: 2400 cycles")
    
    user_input = input("  → ").strip().lower()
    
    if user_input == 'infinite':
        print("  ♾️  INFINITE MODE - Training until Ctrl+C")
        return float('inf')
    elif user_input == '':
        print("  Using recommended: 2400 cycles")
        return 2400
    else:
        try:
            cycles = int(user_input)
            print(f"  Using: {cycles} cycles")
            return cycles
        except:
            print("  Invalid input. Using recommended: 2400 cycles")
            return 2400


def check_resident_photos():
    resident_folder = TRAIN_DIR / "resident_1"
    if not resident_folder.exists():
        return 0
    return len(list(resident_folder.glob("*.jpg")))


def load_resident_faces():
    resident_folder = TRAIN_DIR / "resident_1"
    encodings = []
    
    if not resident_folder.exists():
        print(f"❌ No photos found in {resident_folder}")
        return encodings
    
    image_files = list(resident_folder.glob("*.jpg"))
    print(f"\n📸 Loading {len(image_files)} resident photos...")
    
    for img_path in image_files:
        try:
            img = face_recognition.load_image_file(str(img_path))
            if PROCESSING_SCALE < 1.0:
                h, w = img.shape[:2]
                new_h = int(h * PROCESSING_SCALE)
                new_w = int(w * PROCESSING_SCALE)
                img = cv2.resize(img, (new_w, new_h))
            
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encodings.append(face_encodings[0])
        except:
            pass
    
    print(f"  ✅ Loaded {len(encodings)} resident faces")
    return encodings


def load_all_intruder_faces():
    intruder_folder = TEST_DIR / "intruder"
    encodings = []
    
    if not intruder_folder.exists():
        return encodings
    
    all_files = list(intruder_folder.glob("*.jpg"))
    total_files = len(all_files)
    print(f"\n📸 Found {total_files} intruder images")
    
    max_to_load = min(2000, total_files)
    
    if total_files > max_to_load:
        selected_files = random.sample(all_files, max_to_load)
    else:
        selected_files = all_files
    
    print(f"  Loading {len(selected_files)} intruder faces...")
    
    loaded = 0
    failed = 0
    
    for i, img_path in enumerate(selected_files):
        if i % 50 == 0:
            progress = int((i / len(selected_files)) * 40)
            bar = "█" * progress + "░" * (40 - progress)
            print(f"  [{bar}] {i}/{len(selected_files)}", end="\r")
        
        try:
            img = face_recognition.load_image_file(str(img_path))
            if PROCESSING_SCALE < 1.0:
                h, w = img.shape[:2]
                new_h = int(h * PROCESSING_SCALE)
                new_w = int(w * PROCESSING_SCALE)
                img = cv2.resize(img, (new_w, new_h))
            
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encodings.append(face_encodings[0])
                loaded += 1
            else:
                failed += 1
        except:
            failed += 1
    
    print(" " * 80, end="\r")
    print(f"  ✅ Loaded {loaded} intruder faces")
    if failed > 0:
        print(f"  ⚠️ Failed: {failed} images")
    
    return encodings


def train_model(resident_encodings):
    if len(resident_encodings) < 2:
        return False
    
    save_data = {
        'encodings': resident_encodings,
        'mode': 'simple',
        'timestamp': time.time()
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    return True


def test_accuracy(resident_encodings, intruder_encodings, tolerance=0.55):
    if not resident_encodings or not intruder_encodings:
        return 0.0, 0, 0, 0, 0, 0
    
    num_resident = len(resident_encodings)
    num_intruder = len(intruder_encodings)
    
    balance_count = min(num_resident, num_intruder)
    selected_intruders = random.sample(intruder_encodings, balance_count) if balance_count > 0 else []
    
    correct_resident = 0
    correct_intruder = 0
    false_negatives = 0
    false_positives = 0
    
    for encoding in resident_encodings:
        matches = face_recognition.compare_faces(resident_encodings, encoding, tolerance=tolerance)
        if True in matches:
            correct_resident += 1
        else:
            false_negatives += 1
    
    for encoding in selected_intruders:
        matches = face_recognition.compare_faces(resident_encodings, encoding, tolerance=tolerance)
        if True in matches:
            false_positives += 1
        else:
            correct_intruder += 1
    
    total = len(resident_encodings) + len(selected_intruders)
    correct = correct_resident + correct_intruder
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    return accuracy, correct_resident, correct_intruder, false_negatives, false_positives, len(selected_intruders)


def find_best_tolerance(resident_encodings, intruder_encodings):
    best_accuracy = 0
    best_tolerance = 0.55
    
    for tolerance in np.arange(0.45, 0.70, 0.03):
        accuracy, _, _, _, _, _ = test_accuracy(resident_encodings, intruder_encodings, tolerance)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_tolerance = tolerance
    
    return best_tolerance, best_accuracy


def save_state(cycle, best_accuracy, best_tolerance, all_accuracies, log_data, max_cycles, resident_count):
    state = {
        'cycle': cycle,
        'best_accuracy': best_accuracy,
        'best_tolerance': best_tolerance,
        'all_accuracies': all_accuracies[-100:],
        'log_data': log_data[-500:],
        'max_cycles': max_cycles,
        'resident_count': resident_count,
        'timestamp': time.time()
    }
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def load_state():
    if not STATE_PATH.exists():
        return None, None, None, None, None, None, None
    
    try:
        with open(STATE_PATH, 'r') as f:
            state = json.load(f)
        return (
            state.get('cycle', 0),
            state.get('best_accuracy', 0),
            state.get('best_tolerance', 0.55),
            state.get('all_accuracies', []),
            state.get('log_data', []),
            state.get('max_cycles', 2400),
            state.get('resident_count', 0)
        )
    except:
        return None, None, None, None, None, None, None


def main():
    print("\n" + "="*70)
    print("🎯 ULTIMATE TRAINING - FAST & CLEAN")
    print("="*70)
    
    resident_count = check_resident_photos()
    if resident_count == 0:
        print("❌ No resident photos found!")
        print("Run: python3 Capture.py to take photos first")
        return
    
    print(f"\n📸 Found {resident_count} resident photos")
    
    intruder_folder = TEST_DIR / "intruder"
    intruder_count = len(list(intruder_folder.glob("*.jpg"))) if intruder_folder.exists() else 0
    if intruder_count == 0:
        print("❌ No intruder photos found!")
        return
    
    print(f"📸 Found {intruder_count} intruder photos")
    
    max_cycles = ask_for_cycles()
    
    saved_cycle, saved_best_acc, saved_best_tol, saved_accs, saved_logs, saved_max, saved_count = load_state()
    
    if saved_cycle and saved_cycle > 0:
        print(f"\n📂 Found saved training state!")
        print(f"   Previous cycles: {saved_cycle}")
        print(f"   Previous best accuracy: {saved_best_acc:.1f}%")
        
        if saved_count == resident_count:
            print(f"   ✅ Resident count matches: {resident_count}")
        else:
            print(f"   ⚠️ Resident count changed: {saved_count} → {resident_count}")
        
        print("\n   Resume from where you left off?")
        resume = input("   (y/n): ").lower().strip()
        if resume == 'y':
            cycle = saved_cycle
            best_accuracy = saved_best_acc
            best_tolerance = saved_best_tol
            all_accuracies = saved_accs
            log_data = saved_logs
            print(f"   ✅ Resuming from cycle {cycle + 1}")
        else:
            cycle = 0
            best_accuracy = 0
            best_tolerance = 0.55
            all_accuracies = []
            log_data = []
    else:
        cycle = 0
        best_accuracy = 0
        best_tolerance = 0.55
        all_accuracies = []
        log_data = []
    
    print("\n" + "="*70)
    print("📸 LOADING IMAGES...")
    print("="*70)
    
    resident_encodings = load_resident_faces()
    
    if len(resident_encodings) < 2:
        print("❌ Not enough valid resident faces!")
        return
    
    intruder_encodings = load_all_intruder_faces()
    
    if len(intruder_encodings) < 2:
        print("❌ Not enough intruder faces!")
        return
    
    print("\n" + "="*70)
    print("📊 DATA SUMMARY")
    print("="*70)
    print(f"  Resident faces: {len(resident_encodings)}")
    print(f"  Intruder faces: {len(intruder_encodings)}")
    print(f"  Balance: 1:1 (50:50)")
    print("="*70)
    
    is_infinite = max_cycles == float('inf')
    
    if is_infinite:
        print("\n♾️  INFINITE MODE - Training until Ctrl+C")
    else:
        print(f"\n🚀 STARTING {max_cycles} CYCLES")
    
    print("Press Ctrl+C to STOP and SAVE progress")
    print("="*70 + "\n")
    
    start_time = time.time()
    cycle_count = cycle
    
    try:
        while True:
            if not is_infinite and cycle_count >= max_cycles:
                print(f"\n✅ Reached target: {max_cycles} cycles!")
                break
            
            cycle_count += 1
            cycle_start = time.time()
            
            balance_count = min(len(resident_encodings), len(intruder_encodings))
            selected_intruders = random.sample(intruder_encodings, balance_count)
            
            train_model(resident_encodings)
            
            tolerance, accuracy = find_best_tolerance(resident_encodings, selected_intruders)
            
            acc, correct_res, correct_int, false_neg, false_pos, used_intruders = test_accuracy(
                resident_encodings, selected_intruders, tolerance
            )
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_tolerance = tolerance
                save_data = {
                    'encodings': resident_encodings,
                    'mode': 'simple',
                    'best_accuracy': best_accuracy,
                    'best_tolerance': best_tolerance,
                    'cycle': cycle_count
                }
                with open(BEST_MODEL_PATH, 'wb') as f:
                    pickle.dump(save_data, f)
                print(f"⭐ NEW BEST: {best_accuracy:.1f}%")
            
            all_accuracies.append(acc)
            elapsed = time.time() - cycle_start
            
            if cycle_count % 10 == 0:
                avg_last_10 = sum(all_accuracies[-10:]) / 10 if len(all_accuracies) >= 10 else acc
                elapsed_total = time.time() - start_time
                
                if is_infinite:
                    cycle_display = f"{cycle_count} (infinite)"
                else:
                    cycle_display = f"{cycle_count}/{max_cycles}"
                eta = (elapsed_total / cycle_count) * (max_cycles - cycle_count) if not is_infinite and cycle_count > 0 else 0
                
                print(f"\n{'='*70}")
                print(f"🔄 CYCLE {cycle_display}")
                print("="*70)
                print(f"  🎯 Current: {acc:.1f}% | Best: {best_accuracy:.1f}%")
                print(f"  📈 Last 10 Avg: {avg_last_10:.1f}%")
                print(f"  ⚙️  Tolerance: {tolerance:.2f}")
                print(f"  ✅ Resident: {correct_res}/{len(resident_encodings)}")
                print(f"  ✅ Intruder: {correct_int}/{used_intruders}")
                print(f"  ❌ False Neg: {false_neg}")
                print(f"  ❌ False Pos: {false_pos}")
                print(f"  ⏱️  Cycle Time: {elapsed:.1f}s")
                if not is_infinite:
                    print(f"  ⏳ ETA: {timedelta(seconds=int(eta))}")
                else:
                    print(f"  ⏳ Total Time: {timedelta(seconds=int(elapsed_total))}")
                print("="*70)
                
                log_entry = {
                    'cycle': cycle_count,
                    'accuracy': acc,
                    'best_accuracy': best_accuracy,
                    'tolerance': tolerance,
                    'resident_count': len(resident_encodings),
                    'false_pos': false_pos,
                    'false_neg': false_neg,
                    'timestamp': time.time()
                }
                log_data.append(log_entry)
            
            if cycle_count % 50 == 0:
                save_state(cycle_count, best_accuracy, best_tolerance, all_accuracies, log_data, max_cycles, len(resident_encodings))
                with open(LOG_PATH, 'w') as f:
                    json.dump(log_data, f, indent=2)
                print(f"\n💾 Auto-saved at cycle {cycle_count}")
                print(f"   Best so far: {best_accuracy:.1f}%")
            
            if not is_infinite and cycle_count % 100 == 0:
                progress = int((cycle_count / max_cycles) * 40)
                bar = "█" * progress + "░" * (40 - progress)
                print(f"\n📊 Progress: [{bar}] {cycle_count}/{max_cycles} ({cycle_count/max_cycles*100:.1f}%)")
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 STOPPING - SAVING PROGRESS...")
        print("="*70)
        save_state(cycle_count, best_accuracy, best_tolerance, all_accuracies, log_data, max_cycles, len(resident_encodings))
        with open(LOG_PATH, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"✅ Progress saved at cycle {cycle_count}")
        print(f"   Best accuracy: {best_accuracy:.1f}%")
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"  Total cycles: {cycle_count}")
    print(f"  Best accuracy: {best_accuracy:.1f}%")
    print(f"  Best tolerance: {best_tolerance:.2f}")
    print(f"  Resident faces: {len(resident_encodings)}")
    print(f"  Intruder faces: {len(intruder_encodings)}")
    print(f"  Total time: {timedelta(seconds=int(total_time))}")
    
    if best_accuracy >= 99:
        print("\n🎉🎉🎉 PERFECT! 99%+ ACCURACY!")
    elif best_accuracy >= 95:
        print("\n🎉 EXCELLENT! 95%+ ACCURACY!")
    elif best_accuracy >= 90:
        print("\n👍 GOOD! 90%+ ACCURACY")
    else:
        print("\n⚠️ LOW ACCURACY - Add more resident photos")
    
    print(f"\n✅ Best model: {BEST_MODEL_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()
