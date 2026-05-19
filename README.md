# CIFAR-10 Image Classification with CNN and Transfer Learning

## Overview

This project implements a deep learning image classification system using TensorFlow/Keras on the CIFAR-10 dataset. Two approaches were developed and compared:

- A custom Convolutional Neural Network (CNN) trained from scratch
- A MobileNetV2 transfer learning model pretrained on ImageNet

The project also includes a Streamlit web application for interactive image prediction.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

## Dataset

The CIFAR-10 dataset contains 60,000 colour images across 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Images are 32×32 pixels.

---

## Models Implemented

### 1. Custom CNN
The baseline CNN was developed using:
- Conv2D
- BatchNormalization
- ReLU activation
- MaxPooling
- Dropout regularisation

### 2. MobileNetV2 Transfer Learning
The transfer learning model used:
- MobileNetV2 pretrained on ImageNet
- Frozen feature extractor
- Fine-tuning of upper layers
- Input resizing to 96×96

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Custom CNN | 0.8655 | 0.8650 |
| MobileNetV2 Transfer Learning | 0.8688 | 0.8685 |

The transfer learning model slightly outperformed the baseline CNN.

---

## Features

- CNN training pipeline
- Transfer learning pipeline
- Learning curve visualisation
- Confusion matrix generation
- Classification reports
- Streamlit deployment
- Reproducible workflow

---

## Project Structure

cifar10_streamlit/
├── app.py
├── train.py
├── train_transfer.py
├── evaluate.py
├── requirements.txt
├── model.keras
├── model_transfer.keras
├── artifacts/
└── README.md