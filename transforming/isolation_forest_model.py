#!/usr/bin/env python3
"""
Isolation Forest Model Module

This module provides the machine learning functionality for TED procurement outlier detection.
It handles training, prediction, model serialization, and visualization.

Author: Your Name
Date: May 16, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

class IsolationForestModel:
    """Class for building and using the Isolation Forest model"""
    
    def __init__(self, model_path=None, contamination=0.05, model_dir="models", viz_dir="visualizations"):
        self.model_dir = model_dir
        self.viz_dir = viz_dir
        self.model_path = model_path or os.path.join(model_dir, "isolation_forest_model.pkl")
        self.contamination = contamination
        self.model = None
        self.feature_columns = None
        self.numerical_features = None
        self.categorical_features = None
        
        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
    
    def prepare_features(self, df):
        """
        Prepare features for the isolation forest model
        """
        print("\nPreparing features for outlier detection...")
        
        # Identify numerical and categorical features
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Remove ID columns from features
        id_patterns = ['identifier', 'id', 'code', 'date']
        numerical_features = [col for col in numerical_features
                           if not any(pat in col.lower() for pat in id_patterns)]
        categorical_features = [col for col in categorical_features
                             if not any(pat in col.lower() for pat in id_patterns)]
        
        print(f"Selected {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")
        print(f"Numerical features: {', '.join(numerical_features)}")
        print(f"Categorical features: {', '.join(categorical_features)}")
        
        # Check for missing values
        missing_values = df[numerical_features + categorical_features].isnull().sum()
        features_with_missing = missing_values[missing_values > 0]
        if not features_with_missing.empty:
            print("\nFeatures with missing values:")
            for feature, count in features_with_missing.items():
                print(f" {feature}: {count} missing values ({count/len(df)*100:.2f}%)")
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        return numerical_features, categorical_features
    
    def build_pipeline(self, numerical_features, categorical_features):
        """
        Build a preprocessing and isolation forest pipeline
        """
        print("\nBuilding model pipeline...")
        
        # Numerical preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='drop'
        )
        
        # Create the full pipeline with isolation forest
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('outlier_detector', IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1  # Use all available cores
            ))
        ])
        
        return pipeline
    
    def train(self, df, sample_size=None):
        """
        Train an isolation forest model on the dataset
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame for training
        sample_size : int, optional
            Number of rows to sample for training
        """
        print("\nTraining Isolation Forest model...")
        
        # Sample data if needed
        if sample_size and len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
            print(f"Sampled {len(df_sample)} rows from {len(df)} total rows")
        else:
            df_sample = df
            print(f"Using all {len(df)} available rows for training")
        
        # Prepare features
        numerical_features, categorical_features = self.prepare_features(df_sample)
        
        # Create features dataframe
        X = df_sample[numerical_features + categorical_features].copy()
        
        # Build and train the pipeline
        try:
            pipeline = self.build_pipeline(numerical_features, categorical_features)
            
            # Fit the model
            pipeline.fit(X)
            print("Model training completed successfully.")
            
            self.model = pipeline
            self.feature_columns = X.columns.tolist()
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features
            
            return True
        except Exception as e:
            print(f"Error during model training: {e}")
            
            # Try with only numerical features if there was an error
            print("Attempting to train with only numerical features...")
            try:
                pipeline = self.build_pipeline(numerical_features, [])
                X_num = df_sample[numerical_features].copy()
                pipeline.fit(X_num)
                
                print("Model training with numerical features only completed successfully.")
                
                self.model = pipeline
                self.feature_columns = numerical_features
                self.numerical_features = numerical_features
                self.categorical_features = []
                
                return True
            except Exception as e2:
                print(f"Error during fallback training: {e2}")
                return False
    
    def predict(self, df):
        """
        Use the trained model to detect outliers in the dataset
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset for prediction
        Returns:
        --------
        pd.DataFrame
            DataFrame with added prediction results
        """
        print("\nDetecting outliers...")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        try:
            # Ensure we have all required columns
            missing_columns = [col for col in self.feature_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in dataset: {missing_columns}")
                
                # Add missing columns with default values (0 for numeric columns)
                for col in missing_columns:
                    df[col] = 0
                print(f"Added missing columns with default values")
                
            # Ensure columns are in the right order
            all_feature_columns = [col for col in self.feature_columns if col in df.columns]
            
            # Prepare features
            X = df[all_feature_columns].copy()
            
            # Predict outliers (1: inlier, -1: outlier)
            predictions = self.model.predict(X)
            outliers = predictions == -1
            
            # Get anomaly scores if possible
            try:
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(X)
                elif hasattr(self.model[-1], 'decision_function'):  # For pipeline
                    scores = self.model[-1].decision_function(self.model[:-1].transform(X))
                else:
                    scores = None
            except Exception as e:
                print(f"Warning: Could not compute anomaly scores: {e}")
                scores = None
            
            # Add results to the dataframe
            result_df = df.copy()
            result_df['is_outlier'] = outliers
            
            # Add clear text status
            result_df['outlier_status'] = result_df['is_outlier'].apply(
                lambda x: 'OUTLIER' if x else 'NORMAL'
            )
            
            # Add anomaly scores if available
            if scores is not None:
                result_df['anomaly_score'] = scores.round(4)
            
            # Add timestamp of prediction
            result_df['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Print outlier summary
            outlier_count = outliers.sum()
            print(f"Detected {outlier_count} outliers out of {len(df)} records ({outlier_count/len(df)*100:.2f}%)")
            
            return result_df
        except Exception as e:
            print(f"Error detecting outliers: {e}")
            raise
    
    def save_model(self):
        """
        Save the trained model and feature information
        """
        print(f"\nSaving model to {self.model_path}...")
        
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        
        # Create a package with all necessary components
        model_package = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"Model saved successfully to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """
        Load a previously trained isolation forest model
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        print(f"Loading model from {self.model_path}...")
        
        try:
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.feature_columns = model_package['feature_columns']
            self.numerical_features = model_package['numerical_features']
            self.categorical_features = model_package['categorical_features']
            
            date_trained = model_package.get('date_trained', 'unknown')
            print(f"Model loaded successfully. Trained on: {date_trained}")
            print(f"Features: {len(self.feature_columns)} total features")
            print(f" - {len(self.numerical_features)} numerical features")
            print(f" - {len(self.categorical_features)} categorical features")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def visualize_outliers(self, result_df, output_file=None):
        """
        Skip visualization and just return empty dict
        Parameters:
        -----------
        result_df : pd.DataFrame
            DataFrame with prediction results
        output_file : str, optional
            Path to save the visualizations (not used)
        Returns:
        --------
        dict
            Empty dictionary
        """
        print("\nSkipping outlier visualizations...")
        return {}
    
    def save_predictions(self, result_df, output_file=None):
        """
        Save the prediction results to a CSV file
        Parameters:
        -----------
        result_df : pd.DataFrame
            DataFrame with prediction results
        output_file : str, optional
            Path to save the CSV file
        Returns:
        --------
        str
            Path to the saved file
        """
        # If no specific output file is provided, create one with timestamp
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join("results", f"outliers_{timestamp}.csv")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"\nSaving predictions to {output_file}...")
        try:
            result_df.to_csv(output_file, index=False)
            print(f"Results saved successfully.")
            
            # Print summary
            outlier_count = result_df['is_outlier'].sum()
            total_count = len(result_df)
            print(f"Summary: {outlier_count} outliers detected out of {total_count} records ({outlier_count/total_count*100:.2f}%)")
            
            return output_file
        except Exception as e:
            print(f"Error saving predictions: {e}")
            return None


# If run directly, perform a test train and predict
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='TED Procurement Outlier Detection')
    parser.add_argument('--input', type=str, required=True, help='Path to preprocessed CSV file')
    parser.add_argument('--output', type=str, default='results/outliers.csv', help='Path to output CSV file')
    parser.add_argument('--model', type=str, default='models/isolation_forest_model.pkl', help='Path to save/load model')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for training')
    parser.add_argument('--contamination', type=float, default=0.05, help='Expected proportion of outliers (0.0-0.5)')
    
    args = parser.parse_args()
    
    # Create model instance
    model = IsolationForestModel(
        model_path=args.model,
        contamination=args.contamination
    )
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Training or prediction
    if args.train:
        print("=== Training Mode ===")
        model.train(df, sample_size=args.sample)
        model.save_model()
    
    if args.predict:
        print("=== Prediction Mode ===")
        if not model.model and not args.train:
            model.load_model()
        
        result_df = model.predict(df)
        model.save_predictions(result_df, args.output)