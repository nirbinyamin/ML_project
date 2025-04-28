# MNIST and User Digit Drawing Classification using CNN and SVM
## Introduction
This project focuses on the classification of handwritten digits using two different machine learning approaches: a Convolutional Neural Network (CNN) and a Support Vector Machine (SVM).

The main objective is to compare the performance of these models on the popular MNIST dataset, as well as on custom user-drawn digits collected via a graphical user interface (GUI). The project includes data preprocessing, model training, evaluation using multiple metrics (accuracy, precision, recall, confusion matrix), and visualizations of results.

Through this comparison, the project highlights the strengths and weaknesses of deep learning versus classical machine learning methods for image classification tasks.

## Dataset

The primary dataset used in this project is the **MNIST dataset** (Modified National Institute of Standards and Technology), which contains **70,000 grayscale images** of handwritten digits ranging from 0 to 9. Each image is **28×28 pixels**, centered and normalized for optimal classification performance.

- **Training set:** 60,000 images  
- **Test set:** 10,000 images

During preprocessing:
- The training set is **split into 90% for training and 10% for development (validation)**.
- Images are flattened and normalized using `StandardScaler` to standardize pixel values.

In addition, the project includes a **custom GUI** allowing users to draw digits on a canvas. These user-generated drawings undergo the same preprocessing steps:
- Centering and cropping the digit
- Resizing to 28×28 pixels
- Normalization with the previously saved scaler

This ensures consistent input formatting between the MNIST dataset and the user-drawn digits.

**MNIST Dataset Source:**  
[https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

---
> **Notes:**  
> - For the **CNN**, the full training set (54,000 images) and development set (6,000 images) were used.  
> - For the **SVM**, only a subset of **10,000 images** was used for training to reduce runtime, as SVMs can be computationally intensive on large datasets.
---
