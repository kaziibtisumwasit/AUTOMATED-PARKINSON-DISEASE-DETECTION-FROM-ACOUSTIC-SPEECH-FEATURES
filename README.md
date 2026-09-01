# Parkinson's Disease Detection from Voice Using Machine Learning

A BSc thesis project for detecting Parkinson's Disease (PD) from human voice recordings using acoustic and speech-related features and machine learning classification.

---

## 📌 Project Overview

Parkinson's Disease (PD) is a progressive neurological disorder that can affect motor control as well as speech production.

This project investigates whether acoustic characteristics extracted from voice recordings can be used to distinguish between:

- Healthy Control (HC)
- Parkinson's Disease (PD)

The project follows a machine learning pipeline that converts raw WAV voice recordings into numerical acoustic features and then applies feature selection and multiple machine learning classifiers.

---

## 🎓 Academic Project

**Project Type:** BSc Thesis  
**Domain:** Machine Learning / Speech Processing / Healthcare AI  
**Task:** Binary Classification  
**Classes:**

- HC = 0
- PD = 1

**Input:** Voice recording (.wav)

**Output:** Predicted class

---

# 🧠 Research Objective

The main objective of this research is to investigate whether acoustic characteristics of human speech contain useful information for distinguishing Parkinson's Disease from healthy controls.

The system does not replace clinical diagnosis.

It is a research-oriented machine learning classification system.

---

# 🔬 Complete Pipeline

```text
Raw Voice Recordings
        │
        ▼
Audio Loading
        │
        ├── Librosa
        └── Parselmouth
        │
        ▼
Acoustic Feature Extraction
        │
        ├── Jitter
        ├── Shimmer
        ├── HNR
        ├── Fundamental Frequency (F0)
        ├── Formants
        ├── MFCC
        ├── Delta MFCC
        ├── Spectral Features
        ├── Mel Features
        ├── Chroma
        ├── Pitch
        ├── Rhythm
        └── eGeMAPS
        │
        ▼
Recording-Level Feature CSV
        │
        ▼
Patient-Level Aggregation
        │
        ▼
Feature Matrix X
Target y
        │
        ▼
Feature Selection
        │
        ├── RFECV
        ├── 5-fold evaluation
        └── Feature stability analysis
        │
        ▼
28 Selected Features
        │
        ▼
Machine Learning Classifiers
        │
        ├── Logistic Regression
        ├── Linear SVM
        ├── SGD Classifier
        ├── Decision Tree
        ├── Random Forest
        ├── Extra Trees
        ├── AdaBoost
        ├── Gradient Boosting
        └── XGBoost
        │
        ▼
Performance Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1-score
        └── ROC-AUC
