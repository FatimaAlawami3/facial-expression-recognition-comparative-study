# Facial Expression Recognition with CNN, SVM, and Eigenfaces

## Overview

This project presents a **comparative study of facial expression recognition (FER)** using three different machine learning paradigms:

* **Deep learning** with Convolutional Neural Networks (CNN)
* **Traditional machine learning** with Support Vector Machines (SVM)
* **Dimensionality reduction–based methods** using Eigenfaces (PCA)

In addition, a **hybrid Eigenfaces + CNN model** is explored to study the trade-off between computational efficiency and recognition performance. The experiments are conducted on a large-scale subset of the **AffectNet dataset**, reflecting real-world facial variations.

---

## Problem Statement

Facial Expression Recognition is a key component in affective computing, human–computer interaction, and intelligent systems. However, FER systems face several challenges:

* High intra-class variation (pose, lighting, occlusion)
* High-dimensional image data
* Class imbalance in real-world datasets

This project aims to **systematically compare** classical, hybrid, and deep learning approaches under a unified preprocessing and evaluation pipeline to determine the most effective solution for real-world FER.

---

## Dataset

* **Dataset:** AffectNet (curated subset)
* **Total images:** 28,535 facial images
* **Emotion classes (8):** anger, contempt, disgust, fear, happy, neutral, sad, surprise
* **Image preprocessing:**

  * Grayscale conversion
  * Resizing to 128×128 pixels
  * Pixel normalization to [0, 1]

A statistical analysis revealed a **moderately imbalanced dataset**, which was considered during evaluation through class-aware metrics.

---

## Models Implemented

### 1. Convolutional Neural Network (CNN)

* Custom CNN architecture with convolutional, pooling, dropout, and dense layers
* Trained from scratch on grayscale images
* Optimized using Adam optimizer and categorical cross-entropy loss
* Achieved the **best overall performance**

**Test accuracy:** ~67%

---

### 2. Support Vector Machine (SVM)

* Linear SVM classifier (one-vs-rest)
* Input features obtained by flattening normalized images
* Hyperparameters optimized using grid search

**Test accuracy:** ~36.5%

SVM struggled with high-dimensional raw pixel data and class imbalance.

---

### 3. Eigenfaces (PCA)

* Principal Component Analysis used for dimensionality reduction
* Images projected into a lower-dimensional eigenspace
* Classification based on distance in feature space

**Test accuracy:** ~37%

Eigenfaces provided computational efficiency but lacked discriminative power for complex facial expressions.

---

### 4. Eigenfaces + CNN (Hybrid Model)

* PCA reduced images to 100 principal components
* Reduced features fed into a compact CNN

**Test accuracy:** ~44.7%

This hybrid approach improved over classical methods while remaining more efficient than a full CNN.

---

## Evaluation Metrics

To ensure fair comparison across imbalanced classes, the following metrics were used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices

CNN achieved the highest macro-averaged F1-score, indicating better generalization across emotion classes.

---

## Project Structure

```
├── notebooks/
│   ├── CNN.ipynb
│   ├── SVM.ipynb
│   └── EIGEN_CNN.ipynb
│
├── report/
│   └── ARTI406_Project_Report.pdf
│
├── README.md
└── requirements.txt
```

---

## Tools and Technologies

* Python
* TensorFlow / Keras
* scikit-learn
* OpenCV
* NumPy, Matplotlib
* Jupyter Notebook

---

## Key Findings

* CNNs outperform traditional and hybrid models for FER on complex datasets
* SVMs are sensitive to high-dimensional image data without strong feature engineering
* Eigenfaces reduce dimensionality but lose fine-grained facial details
* Hybrid approaches offer a balance between efficiency and accuracy

---

## Team

This project was completed as a **university group project** for ARTI406 – Machine Learning.

---

## Future Work

* Incorporating transfer learning (e.g., MobileNet, ResNet)
* Addressing class imbalance with advanced sampling techniques
* Exploring ensemble and attention-based architectures
* Optimizing models for real-time deployment on edge devices
