"""
Campus Network Intrusion Detection System - NSL-KDD Dataset Preprocessing
Author: Shreya Narayanan
Date: December 2025

This script preprocesses the NSL-KDD dataset for intrusion detection analysis.
The dataset contains network traffic data with 41 features for classification
of normal vs attack patterns in university campus network infrastructure.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class NSLKDDPreprocessor:
    """
    A comprehensive preprocessor for the NSL-KDD dataset specifically designed
    for campus network intrusion detection systems.
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor with dataset path.
        
        Args:
            data_path (str): Path to the NSL-KDD dataset file
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        
        # Define column names based on NSL-KDD specification
        self.column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'label', 'difficulty'
        ]
        
        # Categorical features for one-hot encoding
        self.categorical_features = ['protocol_type', 'service', 'flag']
        
        # Attack categories mapping for comprehensive analysis
        self.attack_categories = {
            'normal': 'normal',
            'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
            'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos',
            'processtable': 'dos', 'udpstorm': 'dos',
            'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
            'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
            'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
            'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
            'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
            'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l',
            'xsnoop': 'r2l', 'worm': 'r2l',
            'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
            'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
            'xterm': 'u2r'
        }
        
    def load_data(self):
        """
        Load the NSL-KDD dataset from the specified path.
        
        Returns:
            pd.DataFrame: Loaded dataset with proper column names
        """
        print("Loading NSL-KDD dataset...")
        try:
            # Load dataset without headers
            self.df = pd.read_csv(self.data_path, header=None, names=self.column_names)
            print(f"Dataset loaded successfully: {self.df.shape[0]} samples, {self.df.shape[1]} features")
            
            # Display basic information
            print("\nDataset Info:")
            print(f"- Total samples: {len(self.df)}")
            print(f"- Features: {len(self.column_names) - 2}")  # Excluding label and difficulty
            print(f"- Missing values: {self.df.isnull().sum().sum()}")
            
            return self.df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def analyze_data_distribution(self):
        """
        Analyze the distribution of classes in the dataset.
        """
        print("\n" + "="*50)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Label distribution
        print("\nLabel Distribution:")
        label_counts = self.df['label'].value_counts()
        print(label_counts)
        
        # Binary classification distribution (Normal vs Attack)
        binary_labels = self.df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
        print("\nBinary Classification Distribution:")
        binary_counts = binary_labels.value_counts()
        print(binary_counts)
        print(f"Normal: {binary_counts['normal']/len(self.df)*100:.2f}%")
        print(f"Attack: {binary_counts['attack']/len(self.df)*100:.2f}%")
        
        # Categorical features analysis
        print("\nCategorical Features Analysis:")
        for feature in self.categorical_features:
            unique_values = self.df[feature].nunique()
            print(f"- {feature}: {unique_values} unique values")
    
    def preprocess_data(self):
        """
        Comprehensive preprocessing of the NSL-KDD dataset.
        
        Returns:
            tuple: Preprocessed features (X) and target (y)
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Step 1: Drop difficulty column
        print("1. Removing difficulty column...")
        self.df = self.df.drop('difficulty', axis=1)
        
        # Step 2: Handle categorical features with One-Hot Encoding
        print("2. Applying One-Hot Encoding to categorical features...")
        df_processed = self.df.copy()
        
        for feature in self.categorical_features:
            print(f"   - Encoding {feature} ({self.df[feature].nunique()} categories)")
            # Create dummy variables
            dummies = pd.get_dummies(self.df[feature], prefix=feature, drop_first=False)
            # Add to processed dataframe
            df_processed = pd.concat([df_processed, dummies], axis=1)
            # Drop original categorical column
            df_processed = df_processed.drop(feature, axis=1)
        
        # Step 3: Binary label encoding (Normal = 0, Attack = 1)
        print("3. Converting labels to binary classification...")
        df_processed['label_binary'] = df_processed['label'].apply(
            lambda x: 0 if x == 'normal' else 1
        )
        
        # Separate features and target
        X = df_processed.drop(['label', 'label_binary'], axis=1)
        y = df_processed['label_binary']
        
        # Step 4: Normalize numerical features
        print("4. Applying StandardScaler normalization to numerical features...")
        
        # Identify numerical columns (exclude one-hot encoded columns)
        numerical_columns = []
        for col in X.columns:
            if not any(cat in col for cat in self.categorical_features):
                numerical_columns.append(col)
        
        print(f"   - Normalizing {len(numerical_columns)} numerical features")
        
        # Apply scaling to numerical features only
        X_scaled = X.copy()
        X_scaled[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        print(f"5. Final dataset shape: Features {X_scaled.shape}, Target {y.shape}")
        
        return X_scaled, y
    
    def save_processed_data(self, X, y, output_filename='nsl_kdd_preprocessed.csv'):
        """
        Save the preprocessed dataset to a CSV file.
        
        Args:
            X (pd.DataFrame): Preprocessed features
            y (pd.Series): Target labels
            output_filename (str): Output filename
        """
        print(f"\n6. Saving preprocessed dataset as '{output_filename}'...")
        
        # Combine features and target
        processed_data = pd.concat([X, y], axis=1)
        
        # Save to the Data directory
        output_path = os.path.join(os.path.dirname(self.data_path), output_filename)
        processed_data.to_csv(output_path, index=False)
        
        print(f"   ✓ Dataset saved successfully at: {output_path}")
        print(f"   ✓ File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return output_path
    
    def generate_summary_report(self, X, y):
        """
        Generate a comprehensive summary report of the preprocessing.
        
        Args:
            X (pd.DataFrame): Preprocessed features
            y (pd.Series): Target labels
        """
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY REPORT")
        print("="*60)
        
        print(f"📊 Original Dataset:")
        print(f"   - Samples: {len(self.df)}")
        print(f"   - Original Features: 41")
        print(f"   - Categorical Features Encoded: {len(self.categorical_features)}")
        
        print(f"\n📈 Processed Dataset:")
        print(f"   - Final Features: {X.shape[1]}")
        print(f"   - Feature Expansion: {X.shape[1] - 41} additional features from encoding")
        print(f"   - Target Classes: 2 (Binary Classification)")
        
        print(f"\n🎯 Class Distribution:")
        class_counts = y.value_counts()
        print(f"   - Normal (0): {class_counts[0]:,} ({class_counts[0]/len(y)*100:.1f}%)")
        print(f"   - Attack (1): {class_counts[1]:,} ({class_counts[1]/len(y)*100:.1f}%)")
        
        print(f"\n🔧 Applied Transformations:")
        print(f"   ✓ One-Hot Encoding for categorical features")
        print(f"   ✓ Binary label encoding (Normal=0, Attack=1)")
        print(f"   ✓ StandardScaler normalization for numerical features")
        print(f"   ✓ Removed difficulty column")
        
        print(f"\n📝 Ready for Machine Learning:")
        print(f"   ✓ No missing values")
        print(f"   ✓ All features normalized")
        print(f"   ✓ Categorical features properly encoded")
        print(f"   ✓ Suitable for classification algorithms")
        
        print("\n" + "="*60)


def main():
    """
    Main execution function for NSL-KDD preprocessing.
    """
    print("🏛️  CAMPUS NETWORK INTRUSION DETECTION SYSTEM")
    print("📊 NSL-KDD Dataset Preprocessing")
    print("🎓 Final Year Academic Project")
    print("📅 December 2025\n")
    
    # Define dataset path
    base_path = "/Users/nirdeshjain/Documents/Campus-Network-Intrusion-Detection-System-for-University-Infrastructure"
    dataset_path = os.path.join(base_path, "Data", "KDDTrain+.txt")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print("Please ensure the NSL-KDD dataset is in the correct location.")
        return
    
    # Initialize preprocessor
    preprocessor = NSLKDDPreprocessor(dataset_path)
    
    try:
        # Load and analyze data
        df = preprocessor.load_data()
        if df is None:
            return
        
        # Analyze data distribution
        preprocessor.analyze_data_distribution()
        
        # Preprocess data
        X, y = preprocessor.preprocess_data()
        
        # Save processed data
        output_path = preprocessor.save_processed_data(X, y)
        
        # Generate summary report
        preprocessor.generate_summary_report(X, y)
        
        print("\n🎉 Preprocessing completed successfully!")
        print(f"🗂️  Processed dataset available at: {os.path.basename(output_path)}")
        print("🚀 Ready for machine learning model training!")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        print("Please check the dataset format and try again.")


if __name__ == "__main__":
    main()
