#!/usr/bin/env python3
"""
Campus Network IDS - Live Demonstration Script
==============================================

This script provides a focused demonstration of the real-time IDS
specifically designed for final year project viva presentations.

Features:
- Quick setup and execution
- Clear visual output
- Professional presentation format
- Automatic statistics summary
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_ids_simulator import RealTimeIDS
import pandas as pd


def presentation_header():
    """Display professional presentation header"""
    print("\n" + "=" * 70)
    print("🏫 CAMPUS NETWORK INTRUSION DETECTION SYSTEM")
    print("   University Infrastructure Security Solution")
    print("=" * 70)
    print("📅 Final Year Project Demonstration")
    print("🎓 Academic Year 2025-26")
    print("🔒 Real-Time Network Security Monitoring")
    print("=" * 70)


def system_overview():
    """Display system technical overview"""
    print("\n📋 SYSTEM OVERVIEW")
    print("-" * 40)
    print("🔍 Dataset: NSL-KDD (Network Security)")
    print("🤖 Algorithm: Gradient Boosting Classifier")
    print("📊 Features: 122 network traffic attributes")
    print("🎯 Classifications: Normal vs Attack Traffic")
    print("⚡ Processing: Real-time packet analysis")
    print("📝 Logging: Automated attack incident reports")


def live_demonstration():
    """Run a focused live demonstration"""
    
    print("\n🚀 LIVE DEMONSTRATION")
    print("=" * 50)
    
    # Initialize system
    model_path = "model_outputs/final_ids_model.pkl"
    data_path = "Data/nsl_kdd_preprocessed.csv"
    log_path = f"viva_demo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Check prerequisites
    if not os.path.exists(model_path):
        print("❌ Model file missing. Please run the training script first.")
        return False
    
    if not os.path.exists(data_path):
        print("❌ Data file missing. Please run the preprocessing script first.")
        return False
    
    # Initialize IDS
    ids = RealTimeIDS(model_path, data_path, log_path)
    
    print("⚙️  Initializing system components...")
    
    if not ids.load_model():
        return False
    if not ids.load_data():
        return False
    
    ids.initialize_attack_log()
    
    print("✅ System ready for demonstration")
    print("\n🔄 Processing network traffic samples...")
    print("-" * 50)
    
    # Demo parameters
    demo_packets = 25
    processing_delay = 0.8  # Slower for presentation visibility
    
    # Process samples
    sample_data = ids.data.sample(demo_packets, random_state=123).reset_index(drop=True)
    
    attack_alerts = []
    normal_count = 0
    
    for i in range(demo_packets):
        sample = sample_data.iloc[i]
        packet_id = i + 1
        
        # Visual processing indicator
        print(f"📡 Analyzing packet {packet_id:2d}/{demo_packets}", end=" ... ")
        time.sleep(processing_delay * 0.3)
        
        # Make prediction
        prediction, confidence = ids.predict_sample(sample)
        ids.total_packets += 1
        
        if prediction == 1:  # Attack detected
            ids.total_attacks += 1
            threat_level = "🔴 HIGH" if confidence > 0.95 else "🟡 MEDIUM"
            print(f"🚨 ATTACK DETECTED! ({threat_level} - {confidence:.1%})")
            
            attack_alerts.append({
                'packet_id': packet_id,
                'confidence': confidence,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            # Log the attack
            ids.log_attack(packet_id, sample, prediction, confidence)
            
            # Brief pause for emphasis
            time.sleep(processing_delay * 0.7)
            
        else:  # Normal traffic
            normal_count += 1
            print("✅ Normal traffic")
            time.sleep(processing_delay * 0.3)
    
    return True, attack_alerts, normal_count


def display_results(attack_alerts, normal_count, total_packets):
    """Display comprehensive results summary"""
    
    print("\n" + "=" * 50)
    print("📊 DEMONSTRATION RESULTS")
    print("=" * 50)
    
    attack_count = len(attack_alerts)
    normal_rate = (normal_count / total_packets) * 100
    attack_rate = (attack_count / total_packets) * 100
    
    print(f"📈 Total Packets Analyzed: {total_packets}")
    print(f"✅ Normal Traffic: {normal_count} ({normal_rate:.1f}%)")
    print(f"🚨 Attacks Detected: {attack_count} ({attack_rate:.1f}%)")
    
    if attack_alerts:
        print(f"\n🔍 Attack Detection Details:")
        print("-" * 30)
        for i, alert in enumerate(attack_alerts[:5], 1):  # Show first 5
            print(f"  {i}. Packet #{alert['packet_id']:2d} at {alert['timestamp']} "
                  f"(Confidence: {alert['confidence']:.1%})")
        
        if len(attack_alerts) > 5:
            print(f"  ... and {len(attack_alerts) - 5} more attacks")
        
        # Average confidence
        avg_confidence = sum(a['confidence'] for a in attack_alerts) / len(attack_alerts)
        print(f"\n📊 Average Attack Detection Confidence: {avg_confidence:.1%}")


def system_capabilities():
    """Highlight system capabilities"""
    print("\n🎯 SYSTEM CAPABILITIES DEMONSTRATED")
    print("-" * 45)
    print("✅ Real-time traffic analysis")
    print("✅ Machine learning-based classification")
    print("✅ High-confidence attack detection")
    print("✅ Automated incident logging")
    print("✅ Statistical performance tracking")
    print("✅ Scalable architecture design")


def technical_achievements():
    """Highlight technical achievements"""
    print("\n🏆 TECHNICAL ACHIEVEMENTS")
    print("-" * 35)
    print("🔸 Data preprocessing pipeline (125K+ samples)")
    print("🔸 Feature engineering (122 network attributes)")
    print("🔸 ML model training & optimization")
    print("🔸 Real-time prediction system")
    print("🔸 Production-ready code architecture")
    print("🔸 Comprehensive testing & validation")


def main():
    """Main presentation function"""
    
    # Clear screen for presentation
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Presentation flow
    presentation_header()
    
    input("\n👆 Press ENTER to begin system overview...")
    system_overview()
    
    input("\n👆 Press ENTER to start live demonstration...")
    
    try:
        result = live_demonstration()
        
        if isinstance(result, tuple):
            success, attack_alerts, normal_count = result
            if success:
                display_results(attack_alerts, normal_count, 25)
            else:
                print("❌ Demonstration failed")
                return
        else:
            print("❌ Demonstration setup failed")
            return
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration stopped")
        return
    
    input("\n👆 Press ENTER to view system capabilities...")
    system_capabilities()
    
    input("\n👆 Press ENTER to view technical achievements...")
    technical_achievements()
    
    print("\n" + "=" * 70)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("   Thank you for your attention!")
    print("=" * 70)
    
    print(f"\n📁 Attack log saved for review:")
    log_files = [f for f in os.listdir('.') if f.startswith('viva_demo_log_')]
    if log_files:
        print(f"   {log_files[-1]}")
    
    print("\n📚 Additional Resources:")
    print("   • PROJECT_OVERVIEW.md - Complete project documentation")
    print("   • REALTIME_IDS_DOCUMENTATION.md - Real-time system details")
    print("   • ML_MODEL_DEVELOPMENT_REPORT.md - Model training report")
    print("   • EDA_ANALYSIS_REPORT.md - Data analysis report")


if __name__ == "__main__":
    main()
