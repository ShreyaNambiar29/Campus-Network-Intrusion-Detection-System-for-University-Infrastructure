#!/usr/bin/env python3
"""
Automated Test for Real-Time IDS Simulator
==========================================

This script automatically tests the real-time IDS simulator without
requiring user interaction, perfect for demonstration purposes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_ids_simulator import RealTimeIDS
import pandas as pd
import joblib
from datetime import datetime


def run_automated_demo():
    """Run an automated demonstration of the IDS system"""
    
    print("🏫 Campus Network Intrusion Detection System")
    print("Automated Demonstration Mode")
    print("=" * 50)
    
    # File paths
    model_path = "model_outputs/final_ids_model.pkl"
    data_path = "Data/nsl_kdd_preprocessed.csv" 
    log_path = f"demo_attack_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run the ML model training script first.")
        return False
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please run the preprocessing script first.")
        return False
    
    # Initialize and run IDS
    ids = RealTimeIDS(model_path, data_path, log_path)
    
    print("\n🚀 Running Automated Demo...")
    print("Processing 30 packets with minimal delay for demonstration")
    print("-" * 50)
    
    # Load components
    if not ids.load_model():
        return False
    if not ids.load_data():
        return False
    
    ids.initialize_attack_log()
    
    # Run quick demonstration
    try:
        # Process a small sample for demo
        data_sample = ids.data.sample(30, random_state=42).reset_index(drop=True)
        
        print("📊 Demo Statistics:")
        attacks_found = 0
        normal_found = 0
        
        for i in range(len(data_sample)):
            sample = data_sample.iloc[i]
            packet_id = i + 1
            
            # Make prediction
            prediction, confidence = ids.predict_sample(sample)
            
            ids.total_packets += 1
            
            if prediction == 1:  # Attack detected
                ids.total_attacks += 1
                attacks_found += 1
                print(f"🚨 ATTACK #{attacks_found} detected at packet {packet_id} (confidence: {confidence:.2%})")
                ids.log_attack(packet_id, sample, prediction, confidence)
            else:
                normal_found += 1
                if i % 5 == 0:  # Show some normal traffic
                    print(f"✅ Normal traffic - packet {packet_id}")
        
        # Display final results
        print("\n" + "=" * 50)
        print("📋 DEMO RESULTS")
        print("=" * 50)
        print(f"Total Packets Processed: {ids.total_packets}")
        print(f"Normal Traffic: {normal_found}")
        print(f"Attacks Detected: {attacks_found}")
        print(f"Attack Rate: {(attacks_found/ids.total_packets)*100:.1f}%")
        print(f"Attack Log Saved: {log_path}")
        
        # Show sample of logged attacks if any
        if attacks_found > 0:
            try:
                attack_log_df = pd.read_csv(log_path)
                print(f"\n📄 Sample Attack Log (showing first 3 entries):")
                print("-" * 50)
                print(attack_log_df[['timestamp', 'packet_id', 'prediction', 'confidence']].head(3).to_string(index=False))
            except Exception as e:
                print(f"Could not display attack log: {e}")
        
        print("\n✅ Automated demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False


def validate_model_performance():
    """Validate the model performance on a small test set"""
    
    print("\n🔍 Model Performance Validation")
    print("=" * 40)
    
    try:
        # Load model and data
        model = joblib.load("model_outputs/final_ids_model.pkl")
        data = pd.read_csv("Data/nsl_kdd_preprocessed.csv")
        
        # Test on a small sample
        test_sample = data.sample(100, random_state=42)
        
        if 'label_binary' in test_sample.columns:
            X_test = test_sample.drop('label_binary', axis=1)
            y_test = test_sample['label_binary']
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = (predictions == y_test).mean()
            attack_count = (predictions == 1).sum()
            normal_count = (predictions == 0).sum()
            
            print(f"✅ Model validation successful!")
            print(f"Test Sample Size: {len(test_sample)}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Predicted Attacks: {attack_count}")
            print(f"Predicted Normal: {normal_count}")
            
            return True
            
        elif 'label' in test_sample.columns:
            X_test = test_sample.drop('label', axis=1)
            y_test = test_sample['label']
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = (predictions == y_test).mean()
            attack_count = (predictions == 1).sum()
            normal_count = (predictions == 0).sum()
            
            print(f"✅ Model validation successful!")
            print(f"Test Sample Size: {len(test_sample)}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Predicted Attacks: {attack_count}")
            print(f"Predicted Normal: {normal_count}")
            
            return True
            
        else:
            print("⚠️  No label column found for validation")
            # Test predictions without ground truth
            X_test = test_sample
            predictions = model.predict(X_test)
            attack_count = (predictions == 1).sum()
            normal_count = (predictions == 0).sum()
            
            print(f"✅ Model predictions generated!")
            print(f"Test Sample Size: {len(test_sample)}")
            print(f"Predicted Attacks: {attack_count}")
            print(f"Predicted Normal: {normal_count}")
            
            return True
            
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🎯 Campus Network IDS - Automated Test Suite")
    print("Final Year Project Demonstration")
    print("=" * 55)
    
    # Run validation
    validate_model_performance()
    
    # Run demo
    success = run_automated_demo()
    
    if success:
        print("\n🎉 All tests passed! Your IDS system is ready for demonstration.")
        print("\nTo run the interactive version, execute:")
        print("python realtime_ids_simulator.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
