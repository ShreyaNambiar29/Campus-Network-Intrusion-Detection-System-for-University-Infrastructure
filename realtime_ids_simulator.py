#!/usr/bin/env python3
"""
Campus Network Intrusion Detection System - Real-Time Simulator
================================================================

This module simulates a real-time intrusion detection system that monitors
network traffic and classifies packets as Normal or Attack using the trained
machine learning model.

Author: Final Year Project
Date: January 2026
Purpose: Academic demonstration of real-time IDS capabilities
"""

import pandas as pd
import numpy as np
import joblib
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class RealTimeIDS:
    """
    Real-Time Intrusion Detection System Simulator
    
    This class implements a simulated real-time IDS that:
    1. Loads a trained ML model
    2. Streams network traffic data
    3. Makes real-time predictions
    4. Logs attacks and displays alerts
    5. Maintains running statistics
    """
    
    def __init__(self, model_path: str, data_path: str, log_path: str = "attack_log.csv"):
        """
        Initialize the Real-Time IDS
        
        Args:
            model_path (str): Path to the trained model file
            data_path (str): Path to the preprocessed dataset
            log_path (str): Path to save attack logs
        """
        self.model_path = model_path
        self.data_path = data_path
        self.log_path = log_path
        
        # Statistics tracking
        self.total_packets = 0
        self.total_attacks = 0
        self.attack_log = []
        
        # Load model and data
        self.model = None
        self.data = None
        self.feature_columns = None
        
        print("🔒 Campus Network Intrusion Detection System")
        print("=" * 50)
        print("Initializing Real-Time IDS...")
        
    def load_model(self) -> bool:
        """Load the trained machine learning model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Error: Model file not found at {self.model_path}")
                return False
                
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def load_data(self) -> bool:
        """Load the preprocessed dataset for simulation"""
        try:
            if not os.path.exists(self.data_path):
                print(f"❌ Error: Data file not found at {self.data_path}")
                return False
                
            self.data = pd.read_csv(self.data_path)
            
            # Separate features and labels
            if 'label_binary' in self.data.columns:
                self.feature_columns = self.data.columns.drop('label_binary')
            elif 'label' in self.data.columns:
                self.feature_columns = self.data.columns.drop('label')
            else:
                self.feature_columns = self.data.columns
                
            print(f"✅ Data loaded successfully: {len(self.data)} samples")
            print(f"📊 Features: {len(self.feature_columns)} columns")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def initialize_attack_log(self):
        """Initialize the attack log CSV file"""
        try:
            # Create log file with headers
            log_headers = ['timestamp', 'packet_id', 'prediction', 'confidence'] + list(self.feature_columns)
            log_df = pd.DataFrame(columns=log_headers)
            log_df.to_csv(self.log_path, index=False)
            print(f"📝 Attack log initialized: {self.log_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize attack log: {str(e)}")
    
    def predict_sample(self, sample: pd.Series) -> Tuple[int, float]:
        """
        Make prediction on a single network traffic sample
        
        Args:
            sample (pd.Series): Network traffic features
            
        Returns:
            Tuple[int, float]: Prediction (0=Normal, 1=Attack) and confidence
        """
        try:
            # Prepare sample for prediction
            X = sample[self.feature_columns].values.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.95 if prediction == 1 else 0.85
                
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"⚠️  Prediction error: {str(e)}")
            return 0, 0.5
    
    def log_attack(self, packet_id: int, sample: pd.Series, prediction: int, confidence: float):
        """Log detected attack to CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare log entry
            log_entry = {
                'timestamp': timestamp,
                'packet_id': packet_id,
                'prediction': prediction,
                'confidence': confidence
            }
            
            # Add feature values
            for col in self.feature_columns:
                log_entry[col] = sample[col] if col in sample.index else 0
            
            # Append to CSV
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(self.log_path, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"⚠️  Logging error: {str(e)}")
    
    def display_alert(self, packet_id: int, confidence: float):
        """Display attack alert on console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n🚨 SECURITY ALERT 🚨")
        print(f"Time: {timestamp}")
        print(f"Packet ID: {packet_id}")
        print(f"Threat Level: {'HIGH' if confidence > 0.8 else 'MEDIUM'}")
        print(f"Confidence: {confidence:.2%}")
        print("Action: Logged to attack_log.csv")
        print("-" * 40)
    
    def display_statistics(self):
        """Display running statistics"""
        detection_rate = (self.total_attacks / self.total_packets * 100) if self.total_packets > 0 else 0
        
        print(f"\r📊 Stats | Packets: {self.total_packets:,} | "
              f"Attacks: {self.total_attacks:,} | "
              f"Detection Rate: {detection_rate:.2f}%", end="", flush=True)
    
    def simulate_realtime_monitoring(self, max_packets: int = 1000, delay: float = 0.5):
        """
        Simulate real-time network monitoring
        
        Args:
            max_packets (int): Maximum number of packets to process
            delay (float): Delay between packet processing (seconds)
        """
        print(f"\n🚀 Starting Real-Time Monitoring...")
        print(f"📈 Processing up to {max_packets:,} packets with {delay}s intervals")
        print("=" * 60)
        
        try:
            # Shuffle data to simulate random traffic
            data_shuffled = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Process packets
            for i in range(min(max_packets, len(data_shuffled))):
                sample = data_shuffled.iloc[i]
                packet_id = i + 1
                
                # Make prediction
                prediction, confidence = self.predict_sample(sample)
                
                # Update statistics
                self.total_packets += 1
                
                if prediction == 1:  # Attack detected
                    self.total_attacks += 1
                    self.display_alert(packet_id, confidence)
                    self.log_attack(packet_id, sample, prediction, confidence)
                
                # Display running statistics
                if i % 10 == 0 or prediction == 1:  # Update display every 10 packets or on attack
                    self.display_statistics()
                
                # Simulate real-time delay
                time.sleep(delay)
                
                # Check for user interrupt
                if i > 0 and i % 100 == 0:
                    try:
                        user_input = input(f"\n\nProcessed {i} packets. Continue? (y/n): ")
                        if user_input.lower() == 'n':
                            break
                    except KeyboardInterrupt:
                        print("\n\n⏹️  Monitoring stopped by user")
                        break
                        
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
        
        self.display_final_summary()
    
    def display_final_summary(self):
        """Display final monitoring summary"""
        print(f"\n\n" + "=" * 60)
        print("📋 MONITORING SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Packets Analyzed: {self.total_packets:,}")
        print(f"Total Attacks Detected: {self.total_attacks:,}")
        
        if self.total_packets > 0:
            detection_rate = (self.total_attacks / self.total_packets) * 100
            normal_rate = 100 - detection_rate
            
            print(f"Attack Detection Rate: {detection_rate:.2f}%")
            print(f"Normal Traffic Rate: {normal_rate:.2f}%")
        
        print(f"Attack Log File: {self.log_path}")
        print(f"Session Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def run_demo(self, max_packets: int = 100, delay: float = 0.3):
        """
        Run a demonstration of the real-time IDS
        
        Args:
            max_packets (int): Number of packets to process in demo
            delay (float): Delay between packets for demo visibility
        """
        print("\n" + "🎯 DEMO MODE - Campus Network IDS" + "\n")
        
        # Initialize system
        if not self.load_model():
            return False
            
        if not self.load_data():
            return False
            
        self.initialize_attack_log()
        
        print(f"\n✨ Demo Configuration:")
        print(f"   • Model: {os.path.basename(self.model_path)}")
        print(f"   • Dataset: {os.path.basename(self.data_path)}")
        print(f"   • Max Packets: {max_packets:,}")
        print(f"   • Delay: {delay}s per packet")
        print(f"   • Attack Log: {self.log_path}")
        
        # Start monitoring
        self.simulate_realtime_monitoring(max_packets, delay)
        
        return True


def main():
    """Main function to run the Real-Time IDS Simulator"""
    print("🏫 Campus Network Intrusion Detection System")
    print("Real-Time Traffic Monitor & Analyzer")
    print("Final Year Project Demonstration\n")
    
    # File paths
    model_path = "model_outputs/final_ids_model.pkl"
    data_path = "Data/nsl_kdd_preprocessed.csv"
    log_path = "attack_log.csv"
    
    # Check if required files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure you have run the ML model training script first.")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure you have run the preprocessing script first.")
        return
    
    # Initialize IDS
    ids = RealTimeIDS(model_path, data_path, log_path)
    
    # Run demonstration
    print("Select monitoring mode:")
    print("1. Quick Demo (50 packets, fast)")
    print("2. Standard Demo (200 packets, moderate)")
    print("3. Extended Demo (500 packets, slow)")
    print("4. Custom Configuration")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            ids.run_demo(max_packets=50, delay=0.1)
        elif choice == '2':
            ids.run_demo(max_packets=200, delay=0.3)
        elif choice == '3':
            ids.run_demo(max_packets=500, delay=0.5)
        elif choice == '4':
            max_packets = int(input("Enter max packets to process: "))
            delay = float(input("Enter delay between packets (seconds): "))
            ids.run_demo(max_packets=max_packets, delay=delay)
        else:
            print("Invalid choice. Running standard demo...")
            ids.run_demo(max_packets=200, delay=0.3)
            
    except (ValueError, KeyboardInterrupt):
        print("\nRunning default demo configuration...")
        ids.run_demo(max_packets=100, delay=0.3)


if __name__ == "__main__":
    main()
