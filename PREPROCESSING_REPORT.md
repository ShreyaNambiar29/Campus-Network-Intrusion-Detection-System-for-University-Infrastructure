# Campus Network Intrusion Detection System - NSL-KDD Dataset Preprocessing

## Project Overview
**Academic Project:** Final Year - Campus Network Intrusion Detection System for University Infrastructure  
**Dataset:** NSL-KDD (Network Security Laboratory - Knowledge Discovery and Data Mining)  
**Date:** December 2025

## Dataset Summary
- **Original Size:** 125,973 samples with 41 features
- **Final Size:** 125,973 samples with 122 features (after preprocessing)
- **Classes:** Binary classification (Normal vs Attack)
  - Normal Traffic: 67,343 samples (53.5%)
  - Attack Traffic: 58,630 samples (46.5%)

## Preprocessing Steps Completed ✅

### 1. Data Loading & Analysis
- ✅ Loaded NSL-KDD training dataset (`KDDTrain+.txt`)
- ✅ Applied proper column names (41 features + label + difficulty)
- ✅ Verified no missing values in the dataset

### 2. Feature Engineering
- ✅ **Removed difficulty column** as required
- ✅ **One-Hot Encoding** applied to categorical features:
  - `protocol_type`: 3 categories → 3 binary features
  - `service`: 70 categories → 70 binary features  
  - `flag`: 11 categories → 11 binary features
- ✅ **Total feature expansion:** 41 → 122 features

### 3. Label Processing
- ✅ **Binary classification conversion:**
  - Normal → 0
  - All attack types → 1
- ✅ Maintains balance: 53.5% Normal, 46.5% Attack

### 4. Numerical Normalization
- ✅ **StandardScaler** applied to 38 numerical features
- ✅ Features normalized to zero mean and unit variance
- ✅ One-hot encoded features preserved as binary (0/1)

### 5. Data Export
- ✅ Saved processed dataset: `Data/nsl_kdd_preprocessed.csv`
- ✅ File size: 152.62 MB
- ✅ Ready for machine learning model training

## Attack Types in Dataset
The dataset contains various attack categories that were converted to binary labels:

### DoS Attacks (Denial of Service)
- neptune, smurf, back, pod, teardrop, land, etc.

### Probe Attacks (Surveillance)
- satan, ipsweep, nmap, portsweep

### R2L Attacks (Remote to Local)
- warezclient, warezmaster, guess_passwd, ftp_write, imap, multihop, phf, spy

### U2R Attacks (User to Root)  
- buffer_overflow, rootkit, loadmodule, perl

## Technical Implementation

### Libraries Used
```python
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: StandardScaler for normalization
- os: File system operations
```

### Key Features of Implementation
1. **Object-Oriented Design:** `NSLKDDPreprocessor` class for modularity
2. **Comprehensive Logging:** Detailed progress reporting
3. **Error Handling:** Robust exception management
4. **Academic Standards:** Clean, readable, and well-documented code
5. **Scalable Architecture:** Easy to extend for additional preprocessing steps

## File Structure
```
Campus-Network-Intrusion-Detection-System-for-University-Infrastructure/
├── Data/
│   ├── KDDTrain+.txt                 # Original training dataset
│   └── nsl_kdd_preprocessed.csv      # ✅ Processed dataset (READY)
├── nsl_kdd_preprocessing.py          # ✅ Main preprocessing script
├── requirements.txt                  # ✅ Python dependencies
├── venv/                            # Virtual environment
└── README.md                        # Project documentation
```

## Next Steps for IDS Development

### 1. Machine Learning Model Training
- **Classification Algorithms to Consider:**
  - Random Forest (excellent for feature importance analysis)
  - Support Vector Machine (SVM) 
  - Neural Networks (Deep Learning approaches)
  - Gradient Boosting (XGBoost, LightGBM)
  - Naive Bayes (baseline model)

### 2. Model Evaluation Framework
- **Metrics to Track:**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Confusion Matrix
  - False Positive Rate (critical for IDS)
  - Detection Rate for different attack types

### 3. Feature Analysis
- Feature importance ranking
- Correlation analysis
- Dimensionality reduction (PCA if needed)

### 4. Real-time Implementation
- Model deployment for live network monitoring
- Alert generation system
- Integration with campus network infrastructure

## Usage Instructions

### Environment Setup
```bash
# Navigate to project directory
cd "/Users/nirdeshjain/Documents/Campus-Network-Intrusion-Detection-System-for-University-Infrastructure"

# Activate virtual environment
source venv/bin/activate

# Run preprocessing (if needed again)
python nsl_kdd_preprocessing.py
```

### Loading Preprocessed Data
```python
import pandas as pd

# Load the preprocessed dataset
df = pd.read_csv('Data/nsl_kdd_preprocessed.csv')

# Separate features and target
X = df.drop('label_binary', axis=1)  # Features
y = df['label_binary']               # Target (0=Normal, 1=Attack)

# Ready for model training!
```

## Dataset Quality Assurance ✅
- ✅ **No missing values**
- ✅ **Balanced dataset** (53.5% vs 46.5%)
- ✅ **Properly normalized** numerical features
- ✅ **Correctly encoded** categorical features
- ✅ **Clean binary labels** for classification
- ✅ **Academic standard** preprocessing pipeline

---

**Status: PREPROCESSING COMPLETE** 🎉  
**Next Phase: Machine Learning Model Development** 🚀

The dataset is now fully prepared and optimized for training intrusion detection models suitable for university campus network security applications.
