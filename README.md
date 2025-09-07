# Naive Bayes Model for IMDB Sentiment Analysis

A **Naive Bayes classifier** implemented in Python to predict the sentiment (positive or negative) of **IMDB user comments**. This model leverages probabilistic learning to classify text based on word frequencies and prior probabilities.

---

## 📌 Overview
This project uses the **Naive Bayes algorithm** to classify movie reviews from the **Large Movie Review Dataset** as either **positive** or **negative**. The model is trained on **25,000 labeled comments** (12,500 positive and 12,500 negative) and evaluates its performance on a separate test set.

📚 **[Naive Bayes Reference](https://uc-r.github.io/naive_bayes)**

---

## 🛠️ Dependencies
The following Python libraries are required:
```python
from __future__ import division
import tokenizer
import math
import numpy as np
import os
import scipy.interpolate
```

---

## 🔧 Model Workflow

### **1. Training Phase**
- **Input**: Labeled training data (25,000 comments).
- **Process**:
  - Tokenize comments into word features.
  - Compute **conditional probabilities** of words given each class (positive/negative).
  - Store prior probabilities of classes and feature likelihoods.

### **2. Prediction Phase**
- For a **new comment**, the model:
  1. Extracts features (words) and creates a feature vector.
  2. Computes the probability of the comment belonging to each class using **Bayes' theorem**:
     \[
     P(\text{Class} \mid \text{Comment}) \propto P(\text{Comment} \mid \text{Class}) \times P(\text{Class})
     \]
  3. Predicts the class with the **higher probability**.

![Naive Bayes Classifier](naive_bayes_icon.png)

---

## 📊 Dataset
- **Large Movie Review Dataset**:
  - **Train Set**: Contains `pos` and `neg` folders with labeled comments.
  - **Test Set**: Used to evaluate model performance after training.

---

## 📝 Implementation Details
- The code includes **detailed comments** to explain each step of the process.
- The model is trained on the **training set** and evaluated on the **test set** to measure accuracy.
