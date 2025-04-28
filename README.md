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

## Model Architectures

### Convolutional Neural Network (CNN)

The CNN model is designed to classify handwritten digits (0-9) from 28x28 grayscale images. Its architecture is as follows:

- **Input**: 28x28 grayscale image (1 channel)
- **Convolutional Layer 1**: 
  - 32 filters, 3x3 kernel, padding=1
  - Activation: ReLU
  - Max Pooling: 2x2
- **Convolutional Layer 2**:
  - 64 filters, 3x3 kernel, padding=1
  - Activation: ReLU
  - Max Pooling: 2x2
- **Flatten**: Converts the output to a vector of size 7x7x64 = 3136
- **Fully Connected Layer 1**:
  - 128 neurons
  - Activation: ReLU
  - Dropout: 25% to prevent overfitting
- **Fully Connected Layer 2 (Output Layer)**:
  - 10 neurons (one per digit class)
  - Activation: Softmax (applied via CrossEntropyLoss)

**Training Details**:
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Early Stopping: Monitored on validation loss with a patience of 5 epochs
- Maximum Epochs: 100

---

### Support Vector Machine (SVM)

The SVM classifier is trained to classify digits using flattened image vectors (28x28 = 784 features). Three different kernels are used:

- **Linear Kernel**
- **Radial Basis Function (RBF) Kernel**
- **Polynomial Kernel**

**Hyperparameter Tuning**:
- Conducted via Grid Search with 5-fold cross-validation.
- Parameter grid:
  - C: [0.1, 1, 10]
  - gamma: ['scale', 0.01, 0.1] (for RBF and Polynomial)
  - degree: [2, 3, 4] (for Polynomial)

The best hyperparameters for each kernel are selected based on validation accuracy.

**Note on 'scale' for gamma**:
The `'scale'` option for the `gamma` parameter in SVM automatically sets `gamma = 1 / (n_features * X.var())`, where `n_features` is the number of features (784 for MNIST) and `X.var()` is the variance of the training data. This helps adapt the gamma value based on the data's distribution, providing a good default for RBF and Polynomial kernels.


## Project Structure

The project is organized into the following main files and directories:
```
├── cnn_model.py # Defines the architecture of the CNN model
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
```
