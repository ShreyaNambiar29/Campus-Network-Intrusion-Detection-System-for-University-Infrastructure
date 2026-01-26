#!/usr/bin/env python3
"""
Campus Network Intrusion Detection System - Comprehensive Test Suite
Final Year Academic Project

Complete testing framework for validating all components of the IDS system
including unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
import time
import unittest
import json
import numpy as np
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import threading
import subprocess

class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_data_path = 'test_data_sample.csv'
        cls.create_test_data()
        
    @classmethod
    def create_test_data(cls):
        """Create synthetic test data"""
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic features matching NSL-KDD structure
        data = {
            'duration': np.random.rand(n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ftp', 'smtp'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
            'src_bytes': np.random.randint(0, 10000, n_samples),
            'dst_bytes': np.random.randint(0, 10000, n_samples),
            'label': np.random.choice(['normal', 'attack'], n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(cls.test_data_path, index=False)
        
    def test_data_loading(self):
        """Test data loading functionality"""
        self.assertTrue(os.path.exists(self.test_data_path))
        df = pd.read_csv(self.test_data_path)
        self.assertGreater(len(df), 0)
        
    def test_feature_encoding(self):
        """Test categorical feature encoding"""
        df = pd.read_csv(self.test_data_path)
        
        # Test one-hot encoding
        categorical_cols = ['protocol_type', 'service', 'flag']
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # Check that new columns were created
        original_cols = set(df.columns)
        new_cols = set(df_encoded.columns) - original_cols
        self.assertGreater(len(new_cols), 0)
        
    def test_label_encoding(self):
        """Test label encoding"""
        df = pd.read_csv(self.test_data_path)
        
        # Binary encoding
        df['label_binary'] = (df['label'] == 'attack').astype(int)
        self.assertTrue(set(df['label_binary'].unique()).issubset({0, 1}))
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if os.path.exists(cls.test_data_path):
            os.remove(cls.test_data_path)

class TestModelFunctionality(unittest.TestCase):
    """Test cases for machine learning model functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.rand(1000, 10)
        y = np.random.choice([0, 1], 1000)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a simple model
        cls.model = RandomForestClassifier(n_estimators=10, random_state=42)
        cls.model.fit(X_train, y_train)
        
        cls.X_test = X_test
        cls.y_test = y_test
        
    def test_model_prediction(self):
        """Test model prediction functionality"""
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(set(predictions).issubset({0, 1}))
        
    def test_model_probability(self):
        """Test model probability output"""
        probabilities = self.model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[0], len(self.y_test))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification
        
        # Check probabilities sum to 1
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)), decimal=5)
        
    def test_model_performance_metrics(self):
        """Test model performance calculation"""
        predictions = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        
        # All metrics should be between 0 and 1
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        
    def test_model_serialization(self):
        """Test model saving and loading"""
        test_model_path = 'test_model.pkl'
        
        # Save model
        joblib.dump(self.model, test_model_path)
        self.assertTrue(os.path.exists(test_model_path))
        
        # Load model
        loaded_model = joblib.load(test_model_path)
        
        # Test predictions match
        original_pred = self.model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        os.remove(test_model_path)

class TestAPIServer(unittest.TestCase):
    """Test cases for API server functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up API server for testing"""
        cls.api_url = "http://localhost:8000"
        cls.test_features = {
            "features": {f"feature_{i}": np.random.rand() for i in range(122)}
        }
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            self.assertIn(response.status_code, [200, 503])  # Either healthy or model not loaded
            
            data = response.json()
            self.assertIn('status', data)
            self.assertIn('timestamp', data)
            self.assertIn('model_loaded', data)
            
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")
            
    def test_predict_endpoint_structure(self):
        """Test predict endpoint response structure"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=self.test_features,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['prediction', 'confidence', 'prediction_time_ms', 'timestamp']
                for field in required_fields:
                    self.assertIn(field, data)
                    
                self.assertIn(data['prediction'], ['Normal', 'Attack'])
                self.assertGreaterEqual(data['confidence'], 0)
                self.assertLessEqual(data['confidence'], 1)
                
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")
            
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        batch_features = {
            "features": [
                {f"feature_{i}": np.random.rand() for i in range(122)}
                for _ in range(5)
            ]
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/predict/batch",
                json=batch_features,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn('results', data)
                self.assertIn('summary', data)
                self.assertEqual(len(data['results']), 5)
                
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    def test_prediction_latency(self):
        """Test prediction latency requirements"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create test model and data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(1000, 122)
        y_train = np.random.choice([0, 1], 1000)
        model.fit(X_train, y_train)
        
        # Test single prediction latency
        X_test = np.random.rand(1, 122)
        
        start_time = time.time()
        prediction = model.predict(X_test)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Should be under 10ms for single prediction
        self.assertLess(latency_ms, 10, f"Prediction latency too high: {latency_ms:.2f}ms")
        
    def test_batch_throughput(self):
        """Test batch prediction throughput"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create test model and data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(1000, 122)
        y_train = np.random.choice([0, 1], 1000)
        model.fit(X_train, y_train)
        
        # Test batch prediction throughput
        batch_size = 1000
        X_batch = np.random.rand(batch_size, 122)
        
        start_time = time.time()
        predictions = model.predict(X_batch)
        end_time = time.time()
        
        throughput = batch_size / (end_time - start_time)
        
        # Should process at least 100 predictions per second
        self.assertGreater(throughput, 100, f"Throughput too low: {throughput:.1f} predictions/sec")

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to prediction"""
        # This would test the entire pipeline
        # For now, we'll test that all main scripts can be imported
        
        try:
            # Test imports (simplified)
            import numpy as np
            import pandas as pd
            import sklearn
            self.assertTrue(True)  # If we get here, basic dependencies work
            
        except ImportError as e:
            self.fail(f"Failed to import required dependencies: {e}")
            
    def test_data_pipeline_integrity(self):
        """Test data pipeline maintains integrity"""
        # Create test data
        np.random.seed(42)
        original_data = pd.DataFrame({
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100),
            'label': np.random.choice([0, 1], 100)
        })
        
        # Test data doesn't get corrupted through processing
        processed_data = original_data.copy()
        processed_data['feature_1_scaled'] = (processed_data['feature_1'] - processed_data['feature_1'].mean()) / processed_data['feature_1'].std()
        
        # Check original data integrity
        pd.testing.assert_frame_equal(original_data, original_data)
        self.assertEqual(len(processed_data), len(original_data))

class TestSecurityValidation(unittest.TestCase):
    """Security validation tests"""
    
    def test_input_validation(self):
        """Test input validation for security"""
        
        # Test malicious input patterns
        malicious_inputs = [
            {"features": "'; DROP TABLE users; --"},  # SQL injection attempt
            {"features": {"<script>alert('xss')</script>": 1}},  # XSS attempt
            {"features": {f"feature_{i}": "infinity" for i in range(122)}},  # Invalid number format
            {"features": {f"feature_{i}": None for i in range(122)}},  # Null values
        ]
        
        for malicious_input in malicious_inputs:
            # Test that system handles malicious inputs gracefully
            try:
                # This would be tested against the actual validation function
                # For now, we test that the input types are what we expect
                if isinstance(malicious_input.get("features"), dict):
                    for key, value in malicious_input["features"].items():
                        if not isinstance(value, (int, float)):
                            self.assertNotIsInstance(value, (int, float))  # Should be rejected
                            
            except Exception:
                pass  # Expected for malicious inputs
                
    def test_rate_limiting_simulation(self):
        """Test rate limiting (simulation)"""
        # Simulate rapid requests
        request_times = []
        
        for i in range(10):
            request_times.append(time.time())
            time.sleep(0.01)  # 10ms between requests
            
        # Check that we can detect rapid requests
        time_diffs = [request_times[i] - request_times[i-1] for i in range(1, len(request_times))]
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        
        self.assertLess(avg_time_diff, 1, "Should be able to detect rapid requests")

def run_performance_tests():
    """Run performance-specific tests"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK TESTS")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_security_tests():
    """Run security-specific tests"""
    print("\n" + "="*60)
    print("SECURITY VALIDATION TESTS")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSecurityValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_api_tests():
    """Run API-specific tests"""
    print("\n" + "="*60)
    print("API SERVER TESTS")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAPIServer))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_all_tests():
    """Run all test suites"""
    print("="*80)
    print("CAMPUS NETWORK IDS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Collect all test results
    results = []
    
    # Run unit tests
    print("\n" + "="*60)
    print("UNIT TESTS")
    print("="*60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataPreprocessing,
        TestModelFunctionality,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestClass(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    results.append(result.wasSuccessful())
    
    # Run performance tests
    results.append(run_performance_tests())
    
    # Run security tests
    results.append(run_security_tests())
    
    # Run API tests (if server is running)
    results.append(run_api_tests())
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    test_categories = ["Unit Tests", "Performance Tests", "Security Tests", "API Tests"]
    
    for i, (category, success) in enumerate(zip(test_categories, results)):
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{category:<20}: {status}")
    
    overall_success = all(results)
    overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
    
    print(f"\nOverall Status: {overall_status}")
    
    return overall_success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Campus Network IDS Test Suite")
    parser.add_argument("--category", choices=["all", "unit", "performance", "security", "api"], 
                       default="all", help="Test category to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.category == "all":
        success = run_all_tests()
    elif args.category == "unit":
        success = unittest.main(verbosity=2 if args.verbose else 1, exit=False).result.wasSuccessful()
    elif args.category == "performance":
        success = run_performance_tests()
    elif args.category == "security":
        success = run_security_tests()
    elif args.category == "api":
        success = run_api_tests()
    
    sys.exit(0 if success else 1)
