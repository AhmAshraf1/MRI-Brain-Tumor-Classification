# 🧠 MRI Brain Tumor Classification NASNetMobile

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/ahmedashrafhelmi/brain-tumor-classification-using-cnn-nasnetmobile)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Model Architecture](#-model-architecture)
- [Data Preprocessing](#-data-preprocessing)
- [Training Process](#-training-process)
- [Results & Performance](#-results--performance)
- [Visualizations](#-visualizations)
- [Installation](#-installation)
- [Future Improvements](#-future-improvements)
- [References](#-references)

---

## 🎯 Project Overview

This project implements a deep learning solution for **brain tumor classification** using Convolutional Neural Networks (CNN) and transfer learning with **NASNetMobile**. The model classifies brain MRI images into multiple categories to assist in medical diagnosis and treatment planning.

### Key Features
- ✅ Multi-class brain tumor classification
- ✅ Transfer learning with NASNetMobile architecture
- ✅ Data augmentation for improved generalization
- ✅ Comprehensive model evaluation with multiple metrics
- ✅ Visualization of training progress and predictions

### Objectives
1. Develop an accurate automated brain tumor classification system
2. Leverage pre-trained NASNetMobile for efficient feature extraction
3. Achieve high accuracy and reliability for clinical assistance
4. Provide interpretable results through visualization

---

## 📊 Dataset Information

### Dataset Source
**Dataset:** Brain tumors 256x256  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256)

### Dataset Structure
```
Brain-Tumors-256x256/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### Tumor Categories

| Category | Description | Clinical Significance |
|----------|-------------|----------------------|
| **Glioma** | Most common primary brain tumor originating from glial cells | Aggressive, requires immediate treatment |
| **Meningioma** | Tumor arising from meninges (protective membranes) | Usually benign, slow-growing |
| **Pituitary** | Tumor in the pituitary gland | Affects hormone regulation |
| **No Tumor** | Normal brain MRI without tumor presence | Normal |

### Dataset Statistics

```
📊 Dataset Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Set:
  • Glioma:       722 images (~29%)
  • Meningioma:   351 images (~14%)
  • No Tumor:      731 images (~29.5%)
  • Pituitary:      675 images (~27%)
  • Total:         2479 images

Validation Set:
  • Glioma:        89 images
  • Meningioma:    43 images
  • No Tumor:      91 images
  • Pituitary:     84 images
  • Total:        307 images

Testing Set:
  • Glioma:        90 images
  • Meningioma:    44 images
  • No Tumor:      91 images
  • Pituitary:     85 images
  • Total:        310 images
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Data Characteristics
- **Image Format:** PNG/JPEG
- **Image Size:** 256 × 256 pixels
- **Color Mode:** RGB (3 channels)
- **Data Split:** ~80% Training, ~10% Validation  ~10% Testing

---

## 🏗️ Model Architecture

### Transfer Learning with NASNetMobile

**NASNetMobile** is a lightweight neural architecture search (NAS) network optimized for mobile devices while maintaining high accuracy. It was discovered through automated neural architecture search on ImageNet.

#### Architecture Highlights
- **Base Model:** NASNetMobile (pre-trained on ImageNet)
- **Input Shape:** (224, 224, 3)
- **Parameters:** ~5.3M trainable parameters
- **Architecture Type:** Transfer Learning

#### Custom Head Architecture

1. **Global Average Pooling:** Reduces spatial dimensions
2. **Dense Layer (256 units):** Feature extraction with ReLU activation
3. **Dropout (0.5):** Regularization to prevent overfitting
4. **Batch Normalization:** Stabilizes training
5. **Dense Layer (128 units):** Additional feature learning
6. **Dropout (0.3):** Additional regularization
7. **Output Layer (4 units):** Softmax activation for multi-class classification


## 🔧 Data Preprocessing

### Preprocessing Pipeline

#### 1. Resizing & Normalization
```python
Target Size: 224 × 224 pixels (NASNetMobile requirement)

# Pixel values normalized to [0, 1]
pixel_values = pixel_values / 255.0
```

#### 2. Data Augmentation (Training Set)
Data augmentation techniques applied to improve model generalization:

| Augmentation | Parameters | Purpose |
|--------------|-----------|---------|
| **Rotation** | ±15 degrees | Handle different scan orientations |
| **Width Shift** | ±10% | Account for positioning variations |
| **Height Shift** | ±10% | Account for positioning variations |
| **Shear** | 0.2 | Handle perspective distortions |
| **Zoom** | ±20% | Scale invariance |
| **Horizontal Flip** | Yes | Mirror symmetry |
| **Fill Mode** | Nearest | Handle boundary pixels |

#### 3. Class Weights
Calculated to handle class imbalance:
```
Class Weights:
  • Glioma:       0.29
  • Meningioma:   0.29
  • No Tumor:     0.14
  • Pituitary:    0.27
```

## 🎓 Training Process

### Training Configuration

```yaml
Optimizer: Adam
  - Learning Rate: 0.001

Loss Function: Sparse Categorical Crossentropy

Metrics: 
  - Accuracy
  - Precision
  - Recall
  - F1-Score

Batch Size: 8
Epochs: 15
Validation Split: 10% of training data
```

### Callbacks & Techniques

#### 1. Early Stopping
```python
Monitor: val_loss
Patience: 8 epochs
Restore Best Weights: True
```

#### 2. Learning Rate Reduction
```python
Monitor: val_loss
Factor: 0.3
Patience: 2 epochs
Min LR: 1e-10
```

#### 3. Model Checkpoint
```python
Save Best Model: True
Monitor: val_loss
```

### Training Strategy

1. **Freeze Base Model:** Initial training with frozen NASNetMobile layers
2. **Feature Extraction:** Train only custom head (15 epochs)
3. **Fine-Tuning:** Unfreeze last 35 layers of NASNetMobile
4. **Full Training:** Train entire model with reduced learning rate to be 0.0001 (25 epochs)

## 📈 Results & Performance

### Overall Performance Metrics

```
╔═══════════════════════════════════════════════════════════╗
║           FINAL MODEL PERFORMANCE SUMMARY                 ║
╠═══════════════════════════════════════════════════════════╣
║  Training Accuracy:        98.3%                          ║
║  Validation Accuracy:      93.8%                          ║
║  Test Accuracy:           95.2%                           ║
║-----------------------------------------------------------║
║  Training Loss:            0.0508                         ║
║  Validation Loss:          0.1714                         ║
║  Test Loss:               0.2374                          ║
╚═══════════════════════════════════════════════════════════╝
```

### Detailed Classification Report

![Classification Report](<images/model report.png>)

### Confusion Matrix Analysis

![confusion matrix](<images/conf matrix.png>)

#### Key Observations:
- Strong diagonal indicates good classification
- Minimal confusion in giloma tumor type
- Highest accuracy on "No Tumor" and "Pituitary" class (98.0%)
- Some confusion between Glioma and Meningioma (5 cases)

## 📊 Visualizations

### 1. Training History

#### Training & Validation Accuracy Over Epochs

![Accuracy Curves](<images/model acc.png>)
#### Training & Validation Loss Over Epochs

![Loss Curves](<images/model loss.png>)


## 💻 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 🔍 Model Interpretation

### Key Findings

1. **High Accuracy Across All Classes**
   - The model achieves >95% accuracy for all tumor types
   - Minimal false positives/negatives critical for medical applications

2. **Strong Generalization**
   - Small gap between training (98%) and test (95%) accuracy
   - Effective data augmentation prevents overfitting

3. **Clinical Relevance**
   - High precision (0.95) reduces false alarms
   - High recall (0.95) ensures tumor detection
   - Balanced performance suitable for screening tool

### Limitations & Considerations
This model is for research/educational purposes only.

1. **Dataset Limitations**: 
   - Limited to 4 classes
   - Single dataset source
   - Imbalanced classes
   - Small size dataset 

2. **Edge Cases**:
   - May struggle with rare tumor variants
   - Performance on low-quality or corrupted images not evaluated
   - Multi-tumor cases not addressed

---

## 🚀 Future Improvements

### Short-term Enhancements

- [ ] **Increase Dataset Size**: Collect more diverse MRI samples
- [ ] **Add More Classes**: Include additional tumor types 
- [ ] **Performance**: Improve Model performance and time

### Long-term Goals

- [ ] **3D MRI Analysis**: Process full 3D MRI volumes instead of 2D slices
- [ ] **Tumor Segmentation**: Add pixel-level tumor boundary detection
- [ ] **Multi-modal Fusion**: Incorporate CT, PET scans alongside MRI
- [ ] **Real-time Deployment**: Create web/mobile application for inference
- [ ] **Clinical Integration**: Develop DICOM compatibility

### Research Directions

1. **Attention Mechanisms**: Implement self-attention for better feature focus
2. **Few-shot Learning**: Handle rare tumor types with limited data
3. **Uncertainty Quantification**: Provide confidence intervals for predictions
4. **Multi-task Learning**: Simultaneously predict tumor type, grade, and size

---

## 📚 References

### Datasets

- **Brain Tumors 256×256**: [Kaggle Dataset](https://www.kaggle.com/datasets/)

### Related Medical Projects

- [GANs for Kidney and Brain Tumors](https://github.com/AhmAshraf1/GANs-Kidney-Brain-Tumors)
- [COVID-19 CT Scan Classification DenseNet PyTorch](https://github.com/AhmAshraf1/COVID-19-CT-Scan-Classification-PyToch-DenseNet)
- [Medical Text Classification using BiLSTM, BiGRU & Conv1D](https://github.com/AhmAshraf1/Medical-Text-Classification)
- [ECG Signal Classification using RNN, GRU & LSTM](https://github.com/AhmAshraf1/ECG-Signal-Classification)
---

## 👥 Contributing

Contributions are welcome! Please follow these guidelines:
---

## ⚖️ Ethical Considerations

### Medical Ethics
- This model is **NOT** approved for clinical diagnosis
- Always consult qualified medical professionals
- Patient privacy must be maintained at all times
- Informed consent required for any medical data usage

### Data Privacy
- All patient identifiable information must be removed
- Comply with HIPAA, GDPR, and local regulations
- Secure storage and transmission of medical images

### Bias & Fairness
- Model trained on limited demographic data
- May not generalize across all populations
- Continuous monitoring for bias required
- Diverse dataset collection recommended

## 📧 Contact & Support

**Kaggle**: [@ahmedashrafhelmi](https://www.kaggle.com/ahmedashrafhelmi)  
**Project Link**: [Brain Tumor Classification Notebook](https://www.kaggle.com/code/ahmedashrafhelmi/brain-tumor-classification-using-cnn-nasnetmobile)