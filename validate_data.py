"""
Quick validation and statistics script for the preprocessed NSL-KDD dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def validate_preprocessed_data():
    """
    Validate the preprocessed NSL-KDD dataset and show key statistics.
    """
    print("🔍 NSL-KDD PREPROCESSED DATA VALIDATION")
    print("=" * 50)
    
    # Load the preprocessed dataset
    data_path = "Data/nsl_kdd_preprocessed.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Preprocessed dataset not found at {data_path}")
        print("Please run the preprocessing script first.")
        return False
    
    # Load data
    print("📊 Loading preprocessed dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Basic validation
    print("\n🔎 BASIC VALIDATION:")
    print(f"   Dataset shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Separate features and target
    if 'label_binary' in df.columns:
        X = df.drop('label_binary', axis=1)
        y = df['label_binary']
    else:
        print("❌ Error: 'label_binary' column not found")
        return False
    
    # Target distribution
    print(f"\n🎯 TARGET DISTRIBUTION:")
    class_counts = y.value_counts().sort_index()
    for label, count in class_counts.items():
        class_name = "Normal" if label == 0 else "Attack"
        percentage = count / len(y) * 100
        print(f"   {class_name} ({label}): {count:,} samples ({percentage:.1f}%)")
    
    # Feature statistics
    print(f"\n📈 FEATURE STATISTICS:")
    print(f"   Total features: {X.shape[1]}")
    
    # Identify feature types
    binary_features = []
    continuous_features = []
    
    for col in X.columns:
        unique_values = X[col].nunique()
        if unique_values == 2 and set(X[col].unique()).issubset({0, 1, 0.0, 1.0}):
            binary_features.append(col)
        else:
            continuous_features.append(col)
    
    print(f"   Binary features (one-hot encoded): {len(binary_features)}")
    print(f"   Continuous features (normalized): {len(continuous_features)}")
    
    # Sample statistics for continuous features
    if continuous_features:
        print(f"\n📊 CONTINUOUS FEATURES STATISTICS (Sample of 5):")
        sample_features = continuous_features[:5]
        stats_df = X[sample_features].describe()
        print(stats_df.round(4))
    
    # Check normalization
    print(f"\n🔄 NORMALIZATION CHECK:")
    if continuous_features:
        sample_means = X[continuous_features[:5]].mean()
        sample_stds = X[continuous_features[:5]].std()
        print("   Sample feature means (should be ~0):")
        for feature, mean in sample_means.items():
            print(f"     {feature}: {mean:.4f}")
        print("   Sample feature std deviations (should be ~1):")
        for feature, std in sample_stds.items():
            print(f"     {feature}: {std:.4f}")
    
    # Ready for train/test split demonstration
    print(f"\n🚀 READY FOR MODEL TRAINING:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Training ratio: {X_train.shape[0]/len(X)*100:.1f}%")
    print(f"   Test ratio: {X_test.shape[0]/len(X)*100:.1f}%")
    
    # Class distribution in splits
    train_dist = y_train.value_counts(normalize=True).sort_index() * 100
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100
    
    print(f"\n   Class distribution preserved:")
    print(f"     Training - Normal: {train_dist[0]:.1f}%, Attack: {train_dist[1]:.1f}%")
    print(f"     Test - Normal: {test_dist[0]:.1f}%, Attack: {test_dist[1]:.1f}%")
    
    print(f"\n✅ VALIDATION COMPLETE - Dataset is ready for ML model training!")
    return True

def show_feature_categories():
    """
    Show the different categories of features in the dataset.
    """
    data_path = "Data/nsl_kdd_preprocessed.csv"
    df = pd.read_csv(data_path)
    X = df.drop('label_binary', axis=1)
    
    print(f"\n📋 FEATURE CATEGORIES:")
    
    # Identify one-hot encoded features
    protocol_features = [col for col in X.columns if col.startswith('protocol_type_')]
    service_features = [col for col in X.columns if col.startswith('service_')]
    flag_features = [col for col in X.columns if col.startswith('flag_')]
    
    # Original numerical features (remaining)
    encoded_features = protocol_features + service_features + flag_features
    numerical_features = [col for col in X.columns if col not in encoded_features]
    
    print(f"   🌐 Protocol Type Features ({len(protocol_features)}): {protocol_features}")
    print(f"   🔧 Service Features ({len(service_features)}): {service_features[:5]}... (showing first 5)")
    print(f"   🏳️  Flag Features ({len(flag_features)}): {flag_features}")
    print(f"   🔢 Numerical Features ({len(numerical_features)}): {numerical_features[:5]}... (showing first 5)")

if __name__ == "__main__":
    success = validate_preprocessed_data()
    if success:
        show_feature_categories()
    else:
        print("❌ Validation failed. Please check the preprocessing script.")
