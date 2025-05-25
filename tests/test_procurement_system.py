#!/usr/bin/env python3
"""
Unit Tests for EU Procurement Monitoring System
File location: tests/test_procurement_system.py

Run tests with:
python -m pytest tests/test_procurement_system.py -v
or
python -m unittest tests.test_procurement_system -v
"""

import unittest
import tempfile
import os
import pandas as pd
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.ted_data_retriever import TEDDataRetriever
from components.ted_data_preprocessor import TEDDataPreprocessor
from components.ted_data_storage import TEDDataStorage
from transforming.isolation_forest_model import IsolationForestModel
from visualization.visualizer import TEDVisualizer
from ted_ml_pipeline import TEDMLPipeline


class TestTEDDataRetriever(unittest.TestCase):
    """Test cases for TED Data Retriever component"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.retriever = TEDDataRetriever(data_dir=self.temp_dir)
    
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_flatten_notice(self):
        """Test flattening of nested notice structure"""
        notice = {
            "notice-id": "123",
            "values": {"total": 1000, "currency": "EUR"},
            "organization": {"name": "Test Org", "country": "DE"},
            "tags": ["urgent", "construction"]
        }
        
        flattened = self.retriever.flatten_notice(notice)
        
        self.assertEqual(flattened["notice-id"], "123")
        self.assertEqual(flattened["values.total"], 1000)
        self.assertEqual(flattened["values.currency"], "EUR")
        self.assertEqual(flattened["organization.name"], "Test Org")
        self.assertEqual(flattened["tags"], "urgent|construction")
    
    @patch('requests.post')
    def test_get_notices_page_success(self, mock_post):
        """Test successful API page retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            "notices": [
                {"notice-identifier": "TED123", "total-value": "1000"},
                {"notice-identifier": "TED124", "total-value": "2000"}
            ]
        })
        mock_post.return_value = mock_response
        
        result = self.retriever.get_notices_page(1, "20250101", "20250131")
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["notice-identifier"], "TED123")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_get_notices_page_failure(self, mock_post):
        """Test API failure handling"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        result = self.retriever.get_notices_page(1, "20250101", "20250131")
        
        self.assertEqual(result, [])
    
    def test_save_to_csv(self):
        """Test CSV saving functionality"""
        df = pd.DataFrame({
            'notice-id': ['TED123', 'TED124'],
            'total-value': [1000, 2000],
            'country': ['DE', 'FR']
        })
        
        output_file = self.retriever.save_to_csv(df, "test_timestamp")
        
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith("ted_notices_test_timestamp.csv"))
        
        # Verify content
        saved_df = pd.read_csv(output_file)
        self.assertEqual(len(saved_df), 2)
        self.assertEqual(saved_df['notice-id'].iloc[0], 'TED123')


class TestTEDDataPreprocessor(unittest.TestCase):
    """Test cases for TED Data Preprocessor component"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Create test CSV file
        self.test_data = pd.DataFrame({
            'notice-identifier': ['TED123', 'TED124', 'TED125'],
            'total-value': ['1,000.50', '2000', 'invalid'],
            'estimated-value-cur-proc': ['EUR 1000', 'SEK 2000', 'USD 3000'],
            'winner-size': ['small|medium', 'large', ''],
            'organisation-country-buyer': ['DE', 'SE', 'FR'],
            'notice-type': ['Contract notice', 'Contract award', 'Contract notice']
        })
        self.input_file = os.path.join(self.temp_dir, "test_input.csv")
        self.test_data.to_csv(self.input_file, index=False)
        
        self.preprocessor = TEDDataPreprocessor(
            input_file=self.input_file,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv_safely(self):
        """Test robust CSV loading"""
        df = self.preprocessor.load_csv_safely(self.input_file)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('notice-identifier', df.columns)
    
    def test_clean_data_for_ml(self):
        """Test data cleaning functionality"""
        df = self.preprocessor.load_data()
        cleaned_df = self.preprocessor.clean_data_for_ml(df)
        
        # Check currency extraction
        self.assertIn('currency', cleaned_df.columns)
        self.assertEqual(cleaned_df['currency'].iloc[0], 'EUR')
        
        # Check value normalization
        if 'total-value-eur' in cleaned_df.columns:
            self.assertIsInstance(cleaned_df['total-value-eur'].iloc[0], (int, float))
    
    def test_prepare_for_ml(self):
        """Test ML preparation"""
        df = self.preprocessor.load_data()
        cleaned_df = self.preprocessor.clean_data_for_ml(df)
        ml_df = self.preprocessor.prepare_for_ml(cleaned_df)
        
        self.assertIsInstance(ml_df, pd.DataFrame)
        self.assertGreater(len(ml_df.columns), 0)
        
        # Check for numerical columns
        numerical_cols = ml_df.select_dtypes(include=['int64', 'float64']).columns
        self.assertGreater(len(numerical_cols), 0)


class TestTEDDataStorage(unittest.TestCase):
    """Test cases for TED Data Storage component"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_db.db")
        self.storage = TEDDataStorage(db_path=self.db_path)
        
        # Create test result dataframe
        self.test_results = pd.DataFrame({
            'notice-identifier': ['TED123', 'TED124', 'TED125'],
            'total-value-eur': [1000.0, 2000.0, 500.0],
            'organisation-country-buyer': ['DE', 'FR', 'IT'],
            'notice-type': ['Contract notice', 'Award', 'Notice'],
            'is_outlier': [False, True, False],
            'anomaly_score': [0.3, 0.8, 0.2]
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database table creation"""
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('model_runs', tables)
            self.assertIn('procurement_results', tables)
    
    def test_store_results(self):
        """Test storing analysis results"""
        run_id = self.storage.store_results(
            model_type="isolation_forest",
            result_df=self.test_results,
            parameters={"contamination": 0.05},
            notes="Test run"
        )
        
        self.assertIsNotNone(run_id)
        
        # Verify stored data
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM model_runs WHERE run_id = ?", (run_id,))
            run_count = cursor.fetchone()[0]
            self.assertEqual(run_count, 1)
            
            cursor.execute("SELECT COUNT(*) FROM procurement_results WHERE run_id = ?", (run_id,))
            result_count = cursor.fetchone()[0]
            self.assertEqual(result_count, 3)
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        # Store some test data first
        self.storage.store_results("isolation_forest", self.test_results)
        
        stats = self.storage.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_runs', stats)
        self.assertIn('total_records', stats)
        self.assertIn('total_outliers', stats)
        self.assertEqual(stats['total_runs'], 1)
        self.assertEqual(stats['total_records'], 3)
        self.assertEqual(stats['total_outliers'], 1)


class TestIsolationForestModel(unittest.TestCase):
    """Test cases for Isolation Forest Model"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.model = IsolationForestModel(
            model_path=self.model_path,
            contamination=0.1
        )
        
        # Create test dataset
        self.test_data = pd.DataFrame({
            'value_eur': [1000, 2000, 1500, 50000, 1200],  # 50000 should be outlier
            'bidder_count': [3, 2, 4, 1, 3],
            'organization_size': [1, 2, 3, 4, 2],  # Changed from country_code
            'notice_type': ['award', 'notice', 'award', 'award', 'notice']
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    
    def test_train_model(self):
        """Test model training"""
        success = self.model.train(self.test_data)
        
        self.assertTrue(success)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.feature_columns)
    
    def test_predict(self):
        """Test prediction functionality"""
        # Train model first
        self.model.train(self.test_data)
        
        # Make predictions
        result_df = self.model.predict(self.test_data)
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('is_outlier', result_df.columns)
        self.assertIn('outlier_status', result_df.columns)
        self.assertEqual(len(result_df), len(self.test_data))
        
        # Check that some outliers are detected
        outlier_count = result_df['is_outlier'].sum()
        self.assertGreater(outlier_count, 0)
    
    def test_save_and_load_model(self):
        """Test model persistence"""
        # Train and save model
        self.model.train(self.test_data)
        success = self.model.save_model()
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.model_path))
        
        # Create new model instance and load
        new_model = IsolationForestModel(model_path=self.model_path)
        load_success = new_model.load_model()
        self.assertTrue(load_success)
        
        # Test that loaded model works
        result_df = new_model.predict(self.test_data)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('is_outlier', result_df.columns)


class TestTEDVisualizer(unittest.TestCase):
    """Test cases for TED Visualizer component"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = TEDVisualizer(output_dir=self.temp_dir)
        
        # Create test data with outliers
        self.test_data = pd.DataFrame({
            'notice-identifier': ['TED123', 'TED124', 'TED125', 'TED126'],
            'total-value-eur': [1000.0, 2000.0, 50000.0, 1500.0],
            'organisation-country-buyer': ['DE', 'FR', 'DE', 'IT'],
            'notice-type': ['Contract notice', 'Award', 'Notice', 'Award'],
            'is_outlier': [False, False, True, False],
            'anomaly_score': [0.3, 0.4, 0.9, 0.2]
        })
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_visualizations(self):
        """Test visualization creation"""
        viz_files = self.visualizer.create_visualizations(
            self.test_data, 
            base_filename="test_analysis"
        )
        
        self.assertIsInstance(viz_files, list)
        self.assertGreater(len(viz_files), 0)
        
        # Check that files were created
        for file_path in viz_files:
            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(file_path.endswith('.png'))
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame()
        viz_files = self.visualizer.create_visualizations(empty_df)
        
        self.assertEqual(viz_files, [])


class TestTEDMLPipeline(unittest.TestCase):
    """Test cases for TED ML Pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = TEDMLPipeline(base_dir=self.temp_dir)
        
        # Create test ML dataset
        self.test_ml_data = pd.DataFrame({
            'total-value-eur': [1000, 2000, 1500, 50000, 1200],
            'bidder-count': [3, 2, 4, 1, 3],
            'notice_is_Contract notice': [1, 0, 1, 0, 1],
            'notice_is_Award': [0, 1, 0, 1, 0]
        })
        
        self.ml_file = os.path.join(self.temp_dir, "data", "processed", "test_ml.csv")
        os.makedirs(os.path.dirname(self.ml_file), exist_ok=True)
        self.test_ml_data.to_csv(self.ml_file, index=False)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsInstance(self.pipeline, TEDMLPipeline)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "data", "raw")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "data", "processed")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "models")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "output")))
    
    def test_train_model(self):
        """Test model training functionality"""
        model_path = self.pipeline.train(input_file=self.ml_file, contamination=0.2)
        
        self.assertIsNotNone(model_path)
        self.assertTrue(os.path.exists(model_path))
        self.assertIsNotNone(self.pipeline.model)


if __name__ == '__main__':
    # Instructions for running tests
    print("=" * 60)
    print("EU PROCUREMENT MONITORING SYSTEM - UNIT TESTS")
    print("=" * 60)
    print("\nTo run these tests:")
    print("1. Save this file as: tests/test_procurement_system.py")
    print("2. Create an empty __init__.py file in the tests/ directory")
    print("3. Install required dependencies:")
    print("   pip install pytest pandas scikit-learn matplotlib seaborn")
    print("\n4. Run tests using one of these commands:")
    print("   python -m pytest tests/test_procurement_system.py -v")
    print("   python -m unittest tests.test_procurement_system -v")
    print("   python tests/test_procurement_system.py")
    print("\nTest Structure:")
    print("- TestTEDDataRetriever: Tests for data retrieval functionality")
    print("- TestTEDDataPreprocessor: Tests for data preprocessing")
    print("- TestTEDDataStorage: Tests for database operations")
    print("- TestIsolationForestModel: Tests for machine learning model")
    print("- TestTEDVisualizer: Tests for visualization generation")
    print("- TestTEDMLPipeline: Tests for pipeline orchestration")
    print("=" * 60)
    
    # Run the tests
    unittest.main(verbosity=2)