# Melanoma Detection using CNN

## Overview
This project aims to develop a convolutional neural network (CNN) model to detect melanoma accurately. Melanoma is a dangerous form of skin cancer that accounts for 75% of skin cancer-related deaths. An AI-based solution that can evaluate images and alert dermatologists can significantly assist in early diagnosis and reduce manual effort.

## Dataset
The dataset consists of **2,357 images** representing malignant and benign oncological diseases, sourced from the **International Skin Imaging Collaboration (ISIC)**. The images are categorized into the following **9 classes**:
- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

### Dataset Download Link
[Download Here](https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view?usp=sharing)

## Project Pipeline

### 1. Data Understanding & Preparation
- Define paths for **train** and **test** images.
- Resize all images to **180x180 pixels**.

### 2. Dataset Creation
- Split dataset into **training** and **validation** sets.
- Use a batch size of **32**.

### 3. Data Visualization
- Generate sample images to visualize one instance per class.

### 4. Model Building & Training (Baseline Model)
- Build a **custom CNN model** (without transfer learning).
- Rescale images to normalize pixel values between **0 and 1**.
- Choose an appropriate **optimizer** and **loss function**.
- Train the model for **~20 epochs**.
- Analyze overfitting/underfitting trends.

### 5. Data Augmentation
- Apply augmentation strategies to improve model generalization.
- Train an augmented dataset for **~20 epochs**.
- Compare performance before and after augmentation.

### 6. Class Distribution Analysis
- Identify class imbalances.
- Determine dominant and underrepresented classes.

### 7. Handling Class Imbalance
- Use the **Augmentor** library to balance classes.
- Retrain the model on a balanced dataset.
- Train for **~30 epochs** and analyze results.

## Model Training & Evaluation
- Evaluate model accuracy, precision, recall, and F1-score.
- Compare baseline model vs. augmented model vs. class-balanced model.
- Identify whether augmentation and class balancing resolved issues.

## Technologies Used
- **TensorFlow/Keras**: Model development and training
- **OpenCV/Pillow**: Image preprocessing
- **Matplotlib/Seaborn**: Data visualization
- **Augmentor**: Data augmentation

## Setup Instructions
1. Install required libraries:
   ```sh
   pip install tensorflow keras opencv-python matplotlib seaborn Augmentor
   ```
2. Download and extract the dataset.
3. Run the **Jupyter Notebook** to execute data preprocessing and model training.
4. Evaluate model performance on the test dataset.

## Findings & Observations
- Baseline model performance.
- Effectiveness of data augmentation.
- Impact of handling class imbalances.

