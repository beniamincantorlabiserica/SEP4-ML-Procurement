#!/usr/bin/env python3
"""
TED ML Pipeline

This script provides the main entry point for the TED procurement data processing and outlier detection.
It coordinates the different stages of the pipeline: data fetching, preprocessing, model training/prediction.

Author: Your Name
Date: May 16, 2025
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Import pipeline components from correct locations
from components.ted_data_retriever import TEDDataRetriever
from components.ted_data_preprocessor import TEDDataPreprocessor
from transforming.isolation_forest_model import IsolationForestModel

class TEDMLPipeline:
    """Main class for the TED procurement data processing and outlier detection pipeline"""
    
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.raw_data_dir = os.path.join(base_dir, "data")
        self.processed_data_dir = os.path.join(base_dir, "data")
        self.model_dir = os.path.join(base_dir, "models")
        self.output_dir = os.path.join(base_dir, "output")
        self.training_data_file = os.path.join(base_dir, "data", "training_data.csv")
        
        # Ensure directories exist
        for directory in [self.raw_data_dir, self.processed_data_dir, 
                         self.model_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.model = None
    
    def train(self, input_file=None, sample_size=None, contamination=0.05):
        """
        Run the training pipeline:
        1. Use the specified training data file or default to training_data.csv
        2. Train the model
        3. Save the model
        """
        print("=== Starting Training Pipeline ===")
        
        # Step 1: Determine which training file to use
        if input_file:
            # Use the file specified by the user
            training_file = input_file
            print(f"\nUsing specified training file: {training_file}")
        else:
            # Use the default training file
            training_file = self.training_data_file
            print(f"\nUsing default training file: {training_file}")
        
        # Check if the training file exists
        if not os.path.exists(training_file):
            print(f"Error: Training file {training_file} does not exist")
            return None
        
        # Step 2: Train the model
        print("\nStep 2: Training the model...")
        self.model = IsolationForestModel(
            model_path=os.path.join(self.model_dir, "isolation_forest_model.pkl"),
            contamination=contamination
        )
        
        # Load the training data
        print(f"Loading training data from: {training_file}")
        ml_df = pd.read_csv(training_file)
        print(f"Loaded training dataset with {len(ml_df)} rows and {len(ml_df.columns)} columns")
        
        # Train the model
        self.model.train(ml_df, sample_size=sample_size)
        
        # Step 3: Save the model
        print("\nStep 3: Saving trained model...")
        self.model.save_model()
        
        print("\nTraining completed successfully.")
        return self.model.model_path
    
    def predict(self, country=None, start_date=None, end_date=None, max_bid_amount=None):
        """
        Run the prediction pipeline:
        1. Retrieve data from TED API based on filters
        2. Preprocess the data
        3. Load the trained model
        4. Make predictions
        5. Save the results to CSV (no visualizations)
        """
        print("=== Starting Prediction Pipeline ===")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Max bid amount: {max_bid_amount}")
        print(f"Country filter: {country}")
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Retrieve data from TED API
        print("\nStep 1: Retrieving data from TED API...")
        data_retriever = TEDDataRetriever(data_dir=self.raw_data_dir)
        
        # Get data from TED API
        _, raw_data_file = data_retriever.fetch_notices(
            start_date=start_date,
            end_date=end_date,
            max_bid_amount=max_bid_amount,
            country=country,
            max_pages=5
        )
        
        if not raw_data_file:
            print("Error: No data retrieved from API")
            return None
        
        # Step 2: Preprocess the data
        print("\nStep 2: Preprocessing retrieved data...")
        preprocessor = TEDDataPreprocessor(
            input_file=raw_data_file,
            output_dir=self.processed_data_dir
        )
        
        output_files = preprocessor.save_output()
        ml_dataset_file = output_files["ml_dataset"]
        
        # Step 3: Load the trained model
        print("\nStep 3: Loading trained model...")
        self.model = IsolationForestModel(
            model_path=os.path.join(self.model_dir, "isolation_forest_model.pkl")
        )
        
        self.model.load_model()
        
        # Step 4: Making predictions
        print("\nStep 4: Making predictions...")
        ml_df = pd.read_csv(ml_dataset_file)
        result_df = self.model.predict(ml_df)
        
        # Step 5: Save results to CSV (no visualizations)
        print("\nStep 5: Saving results...")
        
        # Save predictions to CSV
        csv_path = os.path.join(self.output_dir, f"outliers_{timestamp}.csv")
        self.model.save_predictions(result_df, csv_path)
                
        print("\nPrediction completed successfully.")
        return csv_path
    
    def evaluate(self, input_file=None):
        """
        Run the evaluation pipeline:
        1. Load the data
        2. Load the trained model
        3. Make predictions and evaluate performance
        """
        print("=== Starting Evaluation Pipeline ===")
        
        # Not implemented yet
        print("Evaluation pipeline not implemented yet.")
        return None


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='TED ML Pipeline for procurement outlier detection')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--input', type=str, help='Path to input CSV file (default: ./data/training_data.csv)')
    train_parser.add_argument('--sample', type=int, help='Sample size for training')
    train_parser.add_argument('--contamination', type=float, default=0.05, 
                            help='Expected proportion of outliers (0.0-0.5)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new data')
    predict_parser.add_argument('--start_date', type=str, help='Start date (YYYYMMDD)')
    predict_parser.add_argument('--end_date', type=str, help='End date (YYYYMMDD)')
    predict_parser.add_argument('--max_bid_amount', type=float, help='Maximum bid amount')
    predict_parser.add_argument('--country', type=str, help='Country code (ISO)')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = TEDMLPipeline()
    
    # Execute command
    if args.command == 'train':
        pipeline.train(
            input_file=args.input,
            sample_size=args.sample,
            contamination=args.contamination
        )
    elif args.command == 'predict':
        pipeline.predict(
            country=args.country,
            start_date=args.start_date,
            end_date=args.end_date,
            max_bid_amount=args.max_bid_amount
        )
    elif args.command == 'evaluate':
        pipeline.evaluate(input_file=args.input)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()