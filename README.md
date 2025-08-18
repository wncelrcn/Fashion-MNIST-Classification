# Fashion-MNIST Classification Project

**Author**: Wince Larcen M. Rivano  
**Course**: CS190-2P - CIS303  
**Institution**: Final Lab Classification Assignment

## 📋 Project Overview

This project implements a comprehensive classification system for the Fashion-MNIST dataset, exploring both classical machine learning and deep learning approaches. The goal is to classify fashion items into 10 distinct categories using various preprocessing techniques and algorithmic implementations to achieve optimal performance.

## 🎯 Objectives

- Perform exploratory data analysis on Fashion-MNIST dataset
- Implement comprehensive data preprocessing pipelines
- Compare performance of classical ML vs deep learning approaches
- Optimize models using various scaling and dimensionality reduction techniques
- Achieve high classification accuracy using F1-macro scoring

## 📊 Dataset Description

**Fashion-MNIST Dataset**:

- **Training Set**: 60,000 samples with 784 features (28×28 pixel images)
- **Test Set**: 10,000 samples with 784 features
- **Classes**: 10 fashion categories
- **Label Distribution**: Balanced across all classes

### Fashion Categories:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings:

- **Data Quality**: No missing values detected across all datasets
- **Class Distribution**: Perfectly balanced with 6,000 samples per class
- **Feature Range**: Pixel values ranging from 0-255 (grayscale images)
- **Data Shape**: Each image represented as 784-dimensional vector (28×28 flattened)

### Visualizations:

- Sample images from each fashion category
- Class distribution analysis showing balanced dataset
- Pixel intensity distributions across different clothing types

## ⚙️ Data Preprocessing

### 1. Feature Scaling Methods:

- **StandardScaler**: Z-score normalization (μ=0, σ=1)
- **MinMaxScaler**: Range normalization (0-1)
- **Raw Features**: No scaling (baseline comparison)

### 2. Dimensionality Reduction:

- **Principal Component Analysis (PCA)**: Reduced to 100 components
- **Variance Threshold**: Removal of low-variance features
- Preserved ~95% of original variance while reducing computational complexity

### 3. Data Augmentation (CNN only):

- **Rotation**: ±15 degrees
- **Translation**: ±10% width/height shifts
- **Zoom**: ±10% scaling
- Enhanced model generalization and robustness

### 4. Train-Validation Split:

- **Classical ML**: 80-20 split with stratification
- **Neural Networks**: 90-10 split for larger training set
- Maintained class balance across all splits

## 🤖 Machine Learning Methods

### Classical Machine Learning Approaches

#### 1. **Logistic Regression**

- **Implementation**: L2 regularization, LBFGS solver
- **Best Performance**: Min-Max Scaled + PCA
- **F1-Macro Score**: 0.8581

#### 2. **Softmax Regression (Multinomial Logistic)**

- **Implementation**: Multinomial classification extension
- **Best Performance**: Min-Max Scaled + PCA
- **F1-Macro Score**: 0.8581

#### 3. **K-Nearest Neighbors (KNN)**

- **Hyperparameter Tuning**: Grid search for optimal k (1-30)
- **Best Configuration**: k=6, Min-Max Scaled + PCA
- **F1-Macro Score**: 0.8665

#### 4. **Random Forest**

- **Hyperparameters**: n_estimators=[50,100], max_depth=[None,20]
- **Best Performance**: Standard Scaled + PCA
- **F1-Macro Score**: 0.8686

#### 5. **Support Vector Machine (Linear)**

- **Hyperparameter Tuning**: C=[0.1, 1, 10]
- **Best Performance**: Min-Max Scaled + PCA
- **F1-Macro Score**: 0.8503

#### 6. **Support Vector Machine (RBF Kernel)**

- **Hyperparameters**: C=[0.01-100], gamma=[1e-4 to 'auto']
- **Best Performance**: Min-Max Scaled + PCA
- **F1-Macro Score**: 0.9050 ⭐ **Best Classical ML Model**

#### 7. **XGBoost**

- **Hyperparameters**: n_estimators=100, max_depth=[3,6], learning_rate=[0.1,0.01]
- **Best Performance**: Min-Max Scaled + PCA
- **F1-Macro Score**: 0.8746

#### 8. **Gaussian Naive Bayes**

- **Hyperparameter Tuning**: var_smoothing optimization
- **Best Performance**: Min-Max Scaled + PCA
- **F1-Macro Score**: 0.7733

### Deep Learning Approaches

#### 1. **Multi-Layer Perceptron (MLP)**

- **Architecture**:
  - Input Layer: 784 neurons
  - Hidden Layer 1: 1024 neurons + ReLU + BatchNorm + Dropout(0.4)
  - Hidden Layer 2: 512 neurons + ReLU + BatchNorm + Dropout(0.3)
  - Hidden Layer 3: 256 neurons + ReLU + Dropout(0.2)
  - Output Layer: 10 neurons + Softmax
- **Optimization**: Adam optimizer (lr=1e-3)
- **Regularization**: Early stopping, learning rate reduction
- **F1-Macro Score**: 0.9072

#### 2. **Convolutional Neural Network (CNN)**

- **Architecture**:
  - Conv2D(32, 3×3) + BatchNorm + MaxPooling(2×2)
  - Conv2D(64, 3×3) + BatchNorm + MaxPooling(2×2)
  - Flatten + Dense(128) + BatchNorm + Dropout(0.3)
  - Dense(10) + Softmax
- **Data Format**: Reshaped to 28×28×1 images
- **Normalization**: Pixel values scaled to [0,1]
- **Data Augmentation**: Applied rotation, shifts, zoom
- **F1-Macro Score**: 0.9273 ⭐ **Best Overall Model**

## 📈 Results Summary

### Performance Ranking (F1-Macro Score):

| Rank | Model            | Configuration     | F1-Score   | Accuracy  |
| ---- | ---------------- | ----------------- | ---------- | --------- |
| 🥇   | **CNN**          | Data Augmentation | **0.9273** | **92.8%** |
| 🥈   | MLP              | Standard Scaling  | 0.9072     | 90.7%     |
| 🥉   | SVM (RBF)        | Min-Max + PCA     | 0.9050     | 90.5%     |
| 4    | XGBoost          | Min-Max + PCA     | 0.8746     | 87.5%     |
| 5    | Random Forest    | Standard + PCA    | 0.8686     | 86.9%     |
| 6    | KNN              | Min-Max + PCA     | 0.8665     | 86.7%     |
| 7    | Logistic/Softmax | Min-Max + PCA     | 0.8581     | 85.9%     |
| 8    | SVM (Linear)     | Min-Max + PCA     | 0.8503     | 85.2%     |
| 9    | Gaussian NB      | Min-Max + PCA     | 0.7733     | 77.3%     |

### Key Performance Insights:

#### **Preprocessing Impact**:

- **Min-Max Scaling**: Performed best for distance-based algorithms (KNN, SVM)
- **Standard Scaling**: Optimal for neural networks and tree-based models
- **PCA**: Significantly improved computational efficiency with minimal performance loss
- **Data Augmentation**: Crucial for CNN generalization (+2-3% improvement)

#### **Algorithm Performance Patterns**:

- **Deep Learning**: Superior performance, especially CNN with spatial structure preservation
- **Kernel Methods**: SVM RBF showed excellent non-linear pattern recognition
- **Ensemble Methods**: XGBoost and Random Forest provided robust, consistent results
- **Linear Methods**: Logistic regression surprisingly competitive given simplicity
- **Instance-Based**: KNN performed well but computationally expensive

## 🎯 Final Conclusions

### Best Model: Convolutional Neural Network

- **Final F1-Score**: 92.73%
- **Key Success Factors**:
  - Spatial feature preservation through convolution
  - Effective data augmentation strategy
  - Appropriate regularization (BatchNorm + Dropout)
  - Adam optimization with learning rate scheduling

### Model Selection Insights:

1. **CNN** excelled by leveraging spatial relationships in image data
2. **SVM RBF** achieved excellent results through non-linear kernel mapping
3. **Deep learning** models generally outperformed classical approaches
4. **Preprocessing** choice significantly impacted model performance

## 💡 Recommendations

### For Production Implementation:

1. **Deploy CNN model** for highest accuracy requirements
2. **Use SVM RBF** for faster inference with acceptable accuracy trade-off
3. **Implement ensemble approach** combining CNN + SVM for robust predictions

### For Further Improvements:

1. **Advanced Architectures**:
   - ResNet/DenseNet for deeper networks
   - Attention mechanisms for focus on discriminative features
2. **Enhanced Data Augmentation**:
   - Mixup/CutMix techniques
   - AutoAugment policies
3. **Advanced Preprocessing**:
   - Histogram equalization
   - Edge detection features
4. **Ensemble Methods**:
   - Stacking CNN + SVM + XGBoost
   - Voting classifiers with diverse models

### Computational Considerations:

- **Real-time Applications**: Use optimized SVM RBF or lightweight CNN
- **Batch Processing**: Leverage full CNN with data augmentation
- **Resource Constraints**: Consider Random Forest or XGBoost alternatives

## 📁 Project Structure

```
Fashion-MNIST-Classification/
│
├── README.md                                          # This documentation
├── Rivano_FinalLabClassification.ipynb               # Classical ML approaches
├── Rivano_FinalLabClassification_NeuralNet.ipynb     # Deep learning approaches
├── X_fashion_mnist_train.csv                         # Training features
├── X_fashion_mnist_test.csv                          # Test features
├── y_fashion_mnist_train.csv                         # Training labels
├── Rivano_FinalLabClassificationResults_SVMRBF.csv   # SVM predictions
└── Rivano_FinalLabClassificationResults_CNN.csv      # CNN predictions
```

## 🛠️ Technical Requirements

```python
# Core Libraries
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0

# Deep Learning
tensorflow>=2.18.0
keras>=3.5.0

# Utilities
jupyter>=1.0.0
```

## 🚀 Usage Instructions

1. **Clone Repository**:

   ```bash
   git clone <repository-url>
   cd Fashion-MNIST-Classification
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Classical ML Pipeline**:

   ```bash
   jupyter notebook Rivano_FinalLabClassification.ipynb
   ```

4. **Run Deep Learning Pipeline**:
   ```bash
   jupyter notebook Rivano_FinalLabClassification_NeuralNet.ipynb
   ```

## 📧 Contact

**Wince Larcen M. Rivano**  
CS190-2P - CIS303  
For questions regarding this implementation, please refer to the course documentation or contact through appropriate academic channels.

---

_This project demonstrates comprehensive machine learning pipeline development, from exploratory analysis through production-ready model deployment, showcasing both classical and modern deep learning approaches to image classification._
