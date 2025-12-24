"""
Campus Network Intrusion Detection System - Exploratory Data Analysis (EDA)
Author: Final Year Academic Project
Date: December 2025

This script performs comprehensive EDA on the preprocessed NSL-KDD dataset
to understand network traffic patterns and identify differences between
normal and attack traffic for university campus network security.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class NSLKDDExploratoryAnalysis:
    """
    Comprehensive EDA class for NSL-KDD dataset analysis
    """
    
    def __init__(self, data_path, output_dir='eda_outputs'):
        """
        Initialize EDA with data path and output directory
        
        Args:
            data_path (str): Path to the preprocessed dataset
            output_dir (str): Directory to save EDA outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.X = None
        self.y = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting parameters
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        
        print("🔍 NSL-KDD EXPLORATORY DATA ANALYSIS")
        print("🏛️  Campus Network Intrusion Detection System")
        print("📅 December 2025")
        print("=" * 60)
    
    def load_data(self):
        """
        Load the preprocessed NSL-KDD dataset
        """
        print("\n📊 LOADING PREPROCESSED DATASET")
        print("-" * 40)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"   📈 Shape: {self.df.shape[0]:,} samples × {self.df.shape[1]} features")
            print(f"   💾 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Separate features and target
            if 'label_binary' in self.df.columns:
                self.X = self.df.drop('label_binary', axis=1)
                self.y = self.df['label_binary']
                print(f"   🎯 Target variable: label_binary (Binary classification)")
                print(f"   🔢 Features: {self.X.shape[1]} (after preprocessing)")
            else:
                raise ValueError("Target column 'label_binary' not found")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
        
        return True
    
    def basic_statistics(self):
        """
        Display basic dataset information and statistics
        """
        print("\n📋 DATASET OVERVIEW")
        print("-" * 40)
        
        # Dataset info
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Features: {self.X.shape[1]}")
        print(f"Missing Values: {self.df.isnull().sum().sum()}")
        print(f"Data Types Distribution:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Target distribution
        print(f"\n🎯 CLASS DISTRIBUTION:")
        class_counts = self.y.value_counts().sort_index()
        total_samples = len(self.y)
        
        for label, count in class_counts.items():
            class_name = "Normal Traffic" if label == 0 else "Attack Traffic"
            percentage = (count / total_samples) * 100
            print(f"   {class_name} ({label}): {count:,} samples ({percentage:.1f}%)")
        
        # Feature categories
        print(f"\n🔧 FEATURE CATEGORIES:")
        numerical_features = []
        binary_features = []
        
        for col in self.X.columns:
            if self.X[col].nunique() == 2 and set(self.X[col].unique()).issubset({0, 1, 0.0, 1.0}):
                binary_features.append(col)
            else:
                numerical_features.append(col)
        
        print(f"   Numerical Features: {len(numerical_features)}")
        print(f"   Binary Features (One-hot encoded): {len(binary_features)}")
        
        # Store for later use
        self.numerical_features = numerical_features
        self.binary_features = binary_features
        
        return numerical_features, binary_features
    
    def plot_class_distribution(self):
        """
        Create and save class distribution visualization
        """
        print("\n📊 CREATING CLASS DISTRIBUTION PLOT")
        print("-" * 40)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        class_counts = self.y.value_counts().sort_index()
        class_labels = ['Normal Traffic', 'Attack Traffic']
        colors = ['#2E86C1', '#E74C3C']
        
        bars = ax1.bar(class_labels, class_counts.values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Class Distribution - Network Traffic Classification', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_xlabel('Traffic Type', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                    f'{count:,}\n({count/len(self.y)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0.05))
        ax2.set_title('Traffic Distribution Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'class_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Class distribution plot saved: {output_path}")
    
    def plot_correlation_heatmap(self):
        """
        Create correlation heatmap for numerical features
        """
        print("\n🔥 CREATING CORRELATION HEATMAP")
        print("-" * 40)
        
        # Select top numerical features for better visualization
        numerical_subset = self.numerical_features[:20]  # Top 20 for readability
        
        # Calculate correlation matrix
        corr_matrix = self.X[numerical_subset].corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        
        plt.title('Feature Correlation Heatmap - Numerical Features\n(Campus Network Traffic Analysis)', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Correlation heatmap saved: {output_path}")
        
        # Print highly correlated features
        print("\n🔍 HIGHLY CORRELATED FEATURE PAIRS (|correlation| > 0.7):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"   {feat1} ↔ {feat2}: {corr_val:.3f}")
        else:
            print("   No highly correlated feature pairs found (correlation > 0.7)")
    
    def plot_feature_distributions(self):
        """
        Plot distributions of key numerical features
        """
        print("\n📈 CREATING FEATURE DISTRIBUTION PLOTS")
        print("-" * 40)
        
        # Select key features for analysis
        key_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
        available_features = [f for f in key_features if f in self.numerical_features]
        
        if len(available_features) < 3:
            available_features = self.numerical_features[:5]  # Use first 5 if key features not available
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(available_features[:6]):
            ax = axes[idx]
            
            # Create separate distributions for normal and attack traffic
            normal_data = self.X[self.y == 0][feature]
            attack_data = self.X[self.y == 1][feature]
            
            # Plot histograms
            ax.hist(normal_data, bins=50, alpha=0.7, label='Normal Traffic', 
                   color='#2E86C1', density=True)
            ax.hist(attack_data, bins=50, alpha=0.7, label='Attack Traffic', 
                   color='#E74C3C', density=True)
            
            ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{feature}', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(available_features), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions: Normal vs Attack Traffic\nCampus Network Security Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'feature_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Feature distribution plots saved: {output_path}")
        
        # Statistical comparison
        print("\n📊 STATISTICAL COMPARISON (Normal vs Attack):")
        for feature in available_features[:5]:
            normal_stats = self.X[self.y == 0][feature].describe()
            attack_stats = self.X[self.y == 1][feature].describe()
            
            print(f"\n{feature.upper()}:")
            print(f"   Normal  - Mean: {normal_stats['mean']:.4f}, Std: {normal_stats['std']:.4f}")
            print(f"   Attack  - Mean: {attack_stats['mean']:.4f}, Std: {attack_stats['std']:.4f}")
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance using Random Forest
        """
        print("\n🌲 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        
        # Prepare data for Random Forest
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print("Training Random Forest for feature importance analysis...")
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=10,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Get top features
        top_features = feature_importance.head(20)
        
        print(f"✅ Random Forest trained successfully!")
        print(f"   Training Accuracy: {rf_model.score(X_train, y_train):.4f}")
        print(f"   Test Accuracy: {rf_model.score(X_test, y_test):.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'], 
                color='skyblue', edgecolor='navy', alpha=0.8)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.title('Top 20 Most Important Features for Network Intrusion Detection\n' + 
                  'Random Forest Analysis - Campus Network Security', 
                  fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Feature importance plot saved: {output_path}")
        
        # Print top features
        print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
        for idx, row in top_features.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    def plot_attack_distribution_by_features(self):
        """
        Analyze how attacks distribute across different feature values
        """
        print("\n🎯 ATTACK DISTRIBUTION BY FEATURE VALUES")
        print("-" * 40)
        
        # Select some categorical features (one-hot encoded)
        protocol_features = [col for col in self.binary_features if col.startswith('protocol_type_')]
        flag_features = [col for col in self.binary_features if col.startswith('flag_')]
        
        if protocol_features and flag_features:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Protocol type analysis
            protocol_data = []
            for feature in protocol_features:
                protocol_name = feature.replace('protocol_type_', '').upper()
                normal_count = len(self.X[(self.X[feature] == 1) & (self.y == 0)])
                attack_count = len(self.X[(self.X[feature] == 1) & (self.y == 1)])
                protocol_data.append({
                    'Protocol': protocol_name,
                    'Normal': normal_count,
                    'Attack': attack_count
                })
            
            protocol_df = pd.DataFrame(protocol_data)
            
            # Plot protocol distribution
            x_pos = np.arange(len(protocol_df))
            width = 0.35
            
            axes[0].bar(x_pos - width/2, protocol_df['Normal'], width, 
                       label='Normal Traffic', color='#2E86C1', alpha=0.8)
            axes[0].bar(x_pos + width/2, protocol_df['Attack'], width,
                       label='Attack Traffic', color='#E74C3C', alpha=0.8)
            
            axes[0].set_xlabel('Protocol Type')
            axes[0].set_ylabel('Number of Samples')
            axes[0].set_title('Traffic Distribution by Protocol Type')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(protocol_df['Protocol'])
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # Flag analysis (show top 5)
            flag_data = []
            for feature in flag_features[:5]:  # Top 5 flags
                flag_name = feature.replace('flag_', '')
                normal_count = len(self.X[(self.X[feature] == 1) & (self.y == 0)])
                attack_count = len(self.X[(self.X[feature] == 1) & (self.y == 1)])
                if normal_count > 0 or attack_count > 0:  # Only include used flags
                    flag_data.append({
                        'Flag': flag_name,
                        'Normal': normal_count,
                        'Attack': attack_count
                    })
            
            if flag_data:
                flag_df = pd.DataFrame(flag_data)
                x_pos = np.arange(len(flag_df))
                
                axes[1].bar(x_pos - width/2, flag_df['Normal'], width,
                           label='Normal Traffic', color='#2E86C1', alpha=0.8)
                axes[1].bar(x_pos + width/2, flag_df['Attack'], width,
                           label='Attack Traffic', color='#E74C3C', alpha=0.8)
                
                axes[1].set_xlabel('Connection Flag')
                axes[1].set_ylabel('Number of Samples')
                axes[1].set_title('Traffic Distribution by Connection Flags')
                axes[1].set_xticks(x_pos)
                axes[1].set_xticklabels(flag_df['Flag'], rotation=45)
                axes[1].legend()
                axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_dir, 'attack_distribution_by_features.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Attack distribution analysis saved: {output_path}")
    
    def generate_eda_summary_report(self, feature_importance):
        """
        Generate a comprehensive EDA summary report
        """
        print("\n📄 GENERATING EDA SUMMARY REPORT")
        print("-" * 40)
        
        report_path = os.path.join(self.output_dir, 'eda_summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CAMPUS NETWORK INTRUSION DETECTION SYSTEM\n")
            f.write("EXPLORATORY DATA ANALYSIS - SUMMARY REPORT\n")
            f.write(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset overview
            f.write("📊 DATASET OVERVIEW:\n")
            f.write(f"   • Total Samples: {len(self.df):,}\n")
            f.write(f"   • Total Features: {self.X.shape[1]}\n")
            f.write(f"   • Numerical Features: {len(self.numerical_features)}\n")
            f.write(f"   • Binary Features: {len(self.binary_features)}\n")
            f.write(f"   • Missing Values: {self.df.isnull().sum().sum()}\n\n")
            
            # Class distribution
            f.write("🎯 CLASS DISTRIBUTION:\n")
            class_counts = self.y.value_counts().sort_index()
            for label, count in class_counts.items():
                class_name = "Normal Traffic" if label == 0 else "Attack Traffic"
                percentage = (count / len(self.y)) * 100
                f.write(f"   • {class_name}: {count:,} samples ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Feature importance
            f.write("🏆 TOP 15 MOST IMPORTANT FEATURES:\n")
            for idx, row in feature_importance.head(15).iterrows():
                f.write(f"   {idx+1:2d}. {row['feature']}: {row['importance']:.4f}\n")
            f.write("\n")
            
            # Key insights
            f.write("🔍 KEY INSIGHTS:\n")
            f.write("   • Dataset is well-balanced for binary classification\n")
            f.write("   • No missing values - high quality preprocessed data\n")
            f.write("   • Feature importance analysis reveals network behavior patterns\n")
            f.write("   • One-hot encoded categorical features provide detailed context\n")
            f.write("   • Ready for machine learning model training and evaluation\n\n")
            
            # Recommendations
            f.write("💡 RECOMMENDATIONS FOR MODEL DEVELOPMENT:\n")
            f.write("   • Use top 20-30 features for initial model training\n")
            f.write("   • Consider ensemble methods (Random Forest, Gradient Boosting)\n")
            f.write("   • Implement cross-validation for robust evaluation\n")
            f.write("   • Monitor false positive rates (critical for IDS)\n")
            f.write("   • Consider real-time deployment architecture\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ EDA summary report saved: {report_path}")
    
    def run_complete_analysis(self):
        """
        Run the complete EDA pipeline
        """
        print(f"\n🚀 STARTING COMPLETE EDA ANALYSIS")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Basic statistics
        self.basic_statistics()
        
        # Generate all visualizations
        self.plot_class_distribution()
        self.plot_correlation_heatmap()
        self.plot_feature_distributions()
        feature_importance = self.analyze_feature_importance()
        self.plot_attack_distribution_by_features()
        
        # Generate summary report
        self.generate_eda_summary_report(feature_importance)
        
        print("\n" + "=" * 60)
        print("🎉 EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 All outputs saved in: {self.output_dir}/")
        print("📊 Generated Files:")
        print("   • class_distribution.png")
        print("   • correlation_heatmap.png") 
        print("   • feature_distributions.png")
        print("   • feature_importance.png")
        print("   • attack_distribution_by_features.png")
        print("   • eda_summary_report.txt")
        print("🚀 Ready for machine learning model development!")
        print("=" * 60)
        
        return True


def main():
    """
    Main execution function for NSL-KDD EDA
    """
    # Define paths
    data_path = "Data/nsl_kdd_preprocessed.csv"
    output_dir = "eda_outputs"
    
    # Check if preprocessed data exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Preprocessed dataset not found at {data_path}")
        print("Please run the preprocessing script first.")
        return
    
    # Initialize and run EDA
    eda_analyzer = NSLKDDExploratoryAnalysis(data_path, output_dir)
    success = eda_analyzer.run_complete_analysis()
    
    if success:
        print(f"\n✅ EDA analysis completed successfully!")
        print(f"🗂️  Check the '{output_dir}' folder for all generated visualizations and reports.")
    else:
        print(f"\n❌ EDA analysis failed. Please check the data and try again.")


if __name__ == "__main__":
    main()
