# Facial Emotion Detection Project

This project focuses on detecting facial emotions from images, specifically identifying emotions like happiness, sadness, and neutrality based on facial photos.

## Modules Overview

### 1. **train_emotion_classifier.py**
   - This script is used to train a CNN-based emotion recognition network. It loads images and their corresponding emotion labels, forming pairs to train the model. The network is trained using **Cross-Entropy Loss**, which is commonly used for classification tasks.

### 2. **single_model.py**
   - This module uses the Haar Cascade Classifier for face detection. The `detectMultiScale` method detects faces within the image and returns the bounding box coordinates (x, y, width, height).
   - After detecting faces, it uses a custom **CNN-based emotion recognition network** (implemented in `my_net/classify.py`). The pre-trained model outputs the emotion label corresponding to each detected face.

### 3. **test_classifier_new.py**
   - This script evaluates the performance of the emotion classifier. It tests the model on a **test dataset** and calculates performance metrics such as **True Positives (TP)**, **False Positives (FP)**, **False Negatives (FN)**, and overall **accuracy**.

## Experimentation with Hyperparameters

To assess the model's performance, we experimented with the following hyperparameters:

- **Learning Rate Adjustments:** We tested various learning rates to identify the optimal one for training the emotion recognition model.
- **Batch Size Variations:** Different batch sizes were tested to evaluate their impact on training efficiency and model accuracy.
- **Dropout:** Dropout layers were added to the network to reduce overfitting and improve the model's generalization ability.
- **Batch Normalization (BN):** Batch normalization layers were introduced to stabilize training and improve the overall performance of the model.
