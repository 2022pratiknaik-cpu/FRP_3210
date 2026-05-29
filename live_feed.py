import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import threading
import queue
import time

speech_queue = queue.Queue()

def speech_worker():
    # Depending on the OS, pyttsx3 might need COM initialization in a new thread
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except ImportError:
        pass

    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

def main():
    # Start the text-to-speech background thread
    threading.Thread(target=speech_worker, daemon=True).start()
    
    last_spoken_label = None
    last_spoken_time = 0

    # Initialize the camera (0 is usually the default built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize detector and classifier
    detector = HandDetector(maxHands=1)
    classifier = Classifier("model/keras_model.h5", "data/labels.txt")
    offset = 20
    imgSize = 80
    labels = ["0", "1", "2", "3", "4", "5", "6"]

    print("Starting live feed. Press 'q' or 'ESC' to exit.")

    while True:
        # Read a frame from the camera
        ret, img = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera.")
            break
            
        imgOutput = img.copy()
        
        # Find hands in the frame
        hands, img = detector.findHands(img)
        
        if hands:
            try:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                
                # We need to make sure the crop coordinates don't go out of bounds
                y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
                
                imgCrop = img[y1:y2, x1:x2]
                
                # Only proceed if we have a valid crop
                if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    aspectRatio = h / w
                    
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    current_label = labels[index]
                    
                    current_time = time.time()
                    if (current_label != last_spoken_label or (current_time - last_spoken_time) > 3.0) and speech_queue.empty():
                        speech_queue.put(current_label)
                        last_spoken_label = current_label
                        last_spoken_time = current_time
                        
                    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                  (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, current_label, (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset),
                                  (x + w+offset, y + h+offset), (255, 0, 255), 4)
                                  
                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)
            except Exception as e:
                # If hand is too close to edge, this prevents the script from crashing
                pass

        # Display the frame in a floating window
        cv2.imshow('Live Feed - Press Q to Exit', imgOutput)
        
        # Wait for 1 ms and check if a key is pressed
        key = cv2.waitKey(1) & 0xFF
        
        # If 'q' or 'ESC' is pressed, exit the loop
        if key == ord('q') or key == 27:
            print("Exiting live feed...")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
