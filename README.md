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
## Project Structure
'''
The project is organized into the following main files and directories:
├──cnn_model.py # Defines the architecture of the CNN model
├── train_cnn.py # Handles CNN training, early stopping, and saving plots/metrics 
├── train_svm.py # Handles SVM training with Grid Search and saving models/plots 
├── test_models.py # Evaluates both CNN and SVM models on the test set, saves metrics and confusion matrices 
├── utilities.py # Contains data preprocessing, visualization functions, metrics saving, and confusion matrix plotting 
├── drawing_utils.py # Provides GUI for drawing digits, saving them, and classifying with CNN and SVM models 
├── ML_Project.py # Main entry point: menu for training, testing, visualizing, and using the GUI 
├── requirements.txt # Python dependencies required for running the project 
├── data/ # Contains raw MNIST IDX files (not included in the repository) 
├── data_numpy/ # Contains preprocessed and normalized datasets as NumPy arrays 
├── Models/ # Saved CNN and SVM models 
├── Plots/ # Generated training and evaluation plots (loss, accuracy, confusion matrices) 
├── Metrics/ # Evaluation metrics (accuracy, precision, recall, etc.) saved as text files 
├── UserDrawings/ # User-drawn digit images and corresponding normalized NumPy arrays
'''
