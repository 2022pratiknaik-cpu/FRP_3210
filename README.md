# Hand Gesture Recognition Project

This project implements real-time hand gesture detection using a webcam feed, combining computer vision techniques for hand tracking and a pretrained deep learning Keras model for classification. This handles dynamic image preprocessing, real-time hand cropping, model prediction, and text-to-speech audio output.

## Directory Structure

The hand gesture recognition project is organized in a modular and systematic manner to ensure clarity, reusability, and ease of experimentation:

```text
Hand_Gesture_Recognition_Project/
|
+-- data/
|   +-- labels.txt
|   +-- classes_mapping.npy
|
+-- model/
|   +-- keras_model.h5
|   +-- lstm_asl_model.keras
|
+-- notebooks/
|   +-- Data_Preprocessing_and_EDA.ipynb
|   +-- Colab_Model_Training_and_Evaluation.ipynb
|   +-- main.ipynb
|   +-- testing.ipynb
|
+-- live_feed.py
+-- final.py
+-- test_model.py
|
+-- README.md
```

## Description

- **`live_feed.py`**: The core execution script placed at the root level of the project. It implements real-time hand gesture detection using the `cvzone` module and a trained Keras model, combining computer vision and deep learning for live classification. This module handles camera initialization, hand tracking, image cropping and resizing, model prediction, and text-to-speech output using pyttsx3.
- **`model/`**: Trained deep learning models are stored here, enabling efficient reuse during live inference and reducing computational overhead.
- **`notebooks/`**: Data preprocessing, exploratory data analysis (EDA), and model training workflows are documented here for reproducibility and performance analysis.
- **`data/`**: Necessary configuration files like the classification labels and class mappings are organized here.

## How to Run

1. Ensure you have the required dependencies installed (OpenCV, cvzone, pyttsx3, numpy, tensorflow).
2. Run the main script:
   ```bash
   python live_feed.py
   ```
3. Press `q` or `ESC` to exit the live camera feed.
