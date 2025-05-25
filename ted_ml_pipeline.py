#!/usr/bin/env python3
"""
EU Procurement Monitoring System - TED ML Pipeline
This script provides the main entry point for the TED procurement data processing and outlier detection.
It offers a simplified architecture focusing on core functionality.

"""
import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime, timedelta
import logging

# Import pipeline components
from components.ted_data_retriever import TEDDataRetriever
from components.ted_data_preprocessor import TEDDataPreprocessor
from components.ted_data_storage import TEDDataStorage
from transforming.isolation_forest_model import IsolationForestModel
from visualization.visualizer import TEDVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TEDMLPipeline:
    """Main class for the EU procurement data processing and outlier detection pipeline"""
    
    def __init__(self, base_dir="."):
        """
        Initialize the pipeline with directory structure
        
        Args:
            base_dir (str): Base directory for all data and output
        """
        self.base_dir = base_dir
        
        # Set up directory structure
        self.directories = {
            "raw_data": os.path.join(base_dir, "data", "raw"),
            "processed_data": os.path.join(base_dir, "data", "processed"),
            "models": os.path.join(base_dir, "models"),
            "output": os.path.join(base_dir, "output"),
            "visualizations": os.path.join(base_dir, "visualizations")
        }
        
        # Create directories if they don't exist
        for directory in self.directories.values():
            os.makedirs(directory, exist_ok=True)
        
        # Default training data file
        self.training_data_file = os.path.join(self.directories["processed_data"], "training_data.csv")
        
        # Initialize components
        self.storage = TEDDataStorage(
            db_path=os.path.join(self.directories["processed_data"], "ted_results.db")
        )
        
        self.visualizer = TEDVisualizer(
            output_dir=self.directories["visualizations"]
        )
        
        self.model = None
        
        logger.info("TEDMLPipeline initialized")
    
    def sync_data(self, days_back=30, country=None, max_pages=5, store=True):
        """
        Synchronize data from TED API
        
        Args:
            days_back (int): Number of days to look back for data
            country (str): Optional country code filter
            max_pages (int): Maximum number of pages to fetch from API
            store (bool): Whether to store data in the database
            
        Returns:
            dict: Dictionary with paths to output files
        """
        logger.info(f"Starting data synchronization (days_back={days_back}, country={country})")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        
        print(f"Synchronizing data from {start_date_str} to {end_date_str}")
        
        try:
            # Step 1: Retrieve data from TED API
            data_retriever = TEDDataRetriever(data_dir=self.directories["raw_data"])
            
            # Fetch data
            df, raw_data_file = data_retriever.fetch_notices(
                start_date=start_date_str,
                end_date=end_date_str,
                country=country,
                max_pages=max_pages
            )
            
            if df.empty or not raw_data_file:
                print("No new data retrieved from API")
                return None
            
            # Step 2: Preprocess data
            print(f"Processing {len(df)} records...")
            
            preprocessor = TEDDataPreprocessor(
                input_file=raw_data_file,
                output_dir=self.directories["processed_data"]
            )
            
            # Process and save data
            output_files = preprocessor.save_output()
            
            # Step 3: Store processed data if requested
            if store and 'ml_dataset' in output_files:
                print("Data ready for model processing and storage")
            
            print(f"Data synchronization completed successfully: {len(df)} records processed")
            
            # Return information about the outputs
            result = {
                "raw_file": raw_data_file,
                "record_count": len(df),
                "processed_files": output_files,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during data synchronization: {e}")
            print(f"Error: {str(e)}")
            return None

    def train(self, input_file=None, sample_size=None, contamination=0.05):
        """
        Train an outlier detection model
        
        Args:
            input_file (str): Optional path to training data file
            sample_size (int): Optional sample size to use for training
            contamination (float): Expected proportion of outliers (0.0-0.5)
            
        Returns:
            str: Path to the saved model file
        """
        logger.info(f"Starting model training (sample_size={sample_size}, contamination={contamination})")
        
        try:
            # Step 1: Determine training file to use
            if input_file:
                training_file = input_file
                print(f"Using specified training file: {training_file}")
            else:
                # Use default training file
                training_file = self.training_data_file
                print(f"Using default training file: {training_file}")
            
            # Check if file exists
            if not os.path.exists(training_file):
                print(f"Error: Training file {training_file} does not exist")
                return None
            
            # Step 2: Load training data
            print("Loading training data...")
            try:
                ml_df = pd.read_csv(training_file)
                print(f"Loaded {len(ml_df)} records with {len(ml_df.columns)} features")
            except Exception as e:
                print(f"Error loading training data: {e}")
                return None
            
            # Step 3: Initialize and train the model
            print("Training Isolation Forest model...")
            self.model = IsolationForestModel(
                model_path=os.path.join(self.directories["models"], "isolation_forest_model.pkl"),
                contamination=contamination
            )
            
            # Train the model
            self.model.train(ml_df, sample_size=sample_size)
            
            # Step 4: Save the trained model
            print("Saving trained model...")
            model_path = self.model.save_model()
            
            print(f"Model training completed successfully: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            print(f"Error: {str(e)}")
            return None
    
    def predict(self, input_file=None, country=None, start_date=None, end_date=None, 
               max_bid_amount=None, visualize=False):
        """
        Run prediction to detect outliers in procurement data
        
        Args:
            input_file (str): Optional path to input file (if not provided, will fetch from API)
            country (str): Optional country code filter (for API fetching)
            start_date (str): Start date in format YYYYMMDD (for API fetching)
            end_date (str): End date in format YYYYMMDD (for API fetching)
            max_bid_amount (float): Optional maximum bid amount filter (for API fetching)
            visualize (bool): Whether to generate visualizations
            
        Returns:
            dict: Dictionary with paths to output files
        """
        logger.info(f"Starting prediction (visualize={visualize})")
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Step 1: Get data (either from file or API)
            if input_file and os.path.exists(input_file):
                print(f"Using provided input file: {input_file}")
                ml_dataset_file = input_file
            else:
                print("Fetching data from TED API...")
                
                if not start_date or not end_date:
                    print("Error: start_date and end_date are required for API data fetching")
                    return None
                
                # Fetch from API
                data_retriever = TEDDataRetriever(data_dir=self.directories["raw_data"])
                
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
                
                # Preprocess the data
                print("Preprocessing data...")
                preprocessor = TEDDataPreprocessor(
                    input_file=raw_data_file,
                    output_dir=self.directories["processed_data"]
                )
                
                output_files = preprocessor.save_output()
                ml_dataset_file = output_files["ml_dataset"]
            
            # Step 2: Load the dataset
            print(f"Loading dataset: {ml_dataset_file}")
            try:
                ml_df = pd.read_csv(ml_dataset_file)
                print(f"Loaded {len(ml_df)} records with {len(ml_df.columns)} features")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return None
            
            # Step 3: Load or create the model
            if not self.model:
                print("Loading trained model...")
                model_path = os.path.join(self.directories["models"], "isolation_forest_model.pkl")
                
                if not os.path.exists(model_path):
                    print("No trained model found. Training new model...")
                    self.train(input_file=ml_dataset_file)
                
                self.model = IsolationForestModel(model_path=model_path)
                self.model.load_model()
            
            # Step 4: Make predictions
            print("Detecting outliers...")
            result_df = self.model.predict(ml_df)
            
            # Step 5: Save results
            results_file = os.path.join(self.directories["output"], f"outliers_{timestamp}.csv")
            
            print(f"Saving results to {results_file}")
            result_df.to_csv(results_file, index=False)
            
            # Count outliers
            if 'is_outlier' in result_df.columns:
                outlier_count = result_df['is_outlier'].sum()
                outlier_pct = outlier_count / len(result_df) * 100
                print(f"Detected {outlier_count} outliers out of {len(result_df)} records ({outlier_pct:.2f}%)")
            
            # Step 6: Generate visualizations if requested
            viz_files = []
            if visualize:
                print("Generating visualizations...")
                viz_files = self.visualizer.create_visualizations(
                    result_df, 
                    base_filename=f"ted_analysis_{timestamp}"
                )
                
                if viz_files:
                    print(f"Generated {len(viz_files)} visualizations")
            
            # Save outliers to storage
            try:
                run_id = self.storage.store_results(
                    model_type="isolation_forest",
                    result_df=result_df,
                    parameters={"contamination": self.model.contamination if self.model else 0.05},
                    notes=f"Prediction run from {start_date or 'file'} to {end_date or 'file'}"
                )
                if run_id:
                    print(f"Results stored in database with run ID: {run_id}")
            except Exception as e:
                logger.warning(f"Could not store results in database: {e}")
            
            # Prepare result information
            result = {
                "results_file": results_file,
                "record_count": len(result_df),
                "outlier_count": outlier_count if 'is_outlier' in result_df.columns else 0,
                "visualizations": viz_files,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print("Prediction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            print(f"Error: {str(e)}")
            return None
    
    def evaluate(self, input_file=None):
        """
        Evaluate model performance
        
        Args:
            input_file (str): Path to evaluation data file
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        logger.info("Starting model evaluation")
        
        if not input_file or not os.path.exists(input_file):
            print(f"Error: Evaluation file {input_file} does not exist")
            return None
        
        try:
            # Step 1: Load the dataset
            print(f"Loading evaluation dataset: {input_file}")
            eval_df = pd.read_csv(input_file)
            
            # Step 2: Load the model if not already loaded
            if not self.model:
                print("Loading model...")
                model_path = os.path.join(self.directories["models"], "isolation_forest_model.pkl")
                
                if not os.path.exists(model_path):
                    print("Error: No trained model found")
                    return None
                
                self.model = IsolationForestModel(model_path=model_path)
                self.model.load_model()
            
            # Step 3: Make predictions
            print("Running prediction for evaluation...")
            result_df = self.model.predict(eval_df)
            
            # Step 4: Calculate metrics
            # Note: For unsupervised outlier detection, we don't have ground truth labels
            # So we'll report basic statistics
            
            outlier_count = result_df['is_outlier'].sum()
            total_count = len(result_df)
            outlier_pct = outlier_count / total_count * 100
            
            # If value column exists, calculate value statistics
            value_col = None
            for col in ['total-value-eur', 'total-value-eur-capped', 'value_eur']:
                if col in result_df.columns:
                    value_col = col
                    break
            
            value_stats = {}
            if value_col:
                normal_avg = result_df[~result_df['is_outlier']][value_col].mean()
                outlier_avg = result_df[result_df['is_outlier']][value_col].mean() if outlier_count > 0 else 0
                
                value_stats = {
                    "normal_avg_value": normal_avg,
                    "outlier_avg_value": outlier_avg,
                    "value_ratio": outlier_avg / normal_avg if normal_avg > 0 else 0
                }
            
            # Prepare evaluation metrics
            metrics = {
                "total_records": total_count,
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_pct,
                "value_statistics": value_stats,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save evaluation results
            eval_file = os.path.join(self.directories["output"], f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(eval_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"Evaluation completed: {outlier_count} outliers detected ({outlier_pct:.2f}%)")
            print(f"Results saved to {eval_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            print(f"Error: {str(e)}")
            return None
    
    def generate_report(self, data_file=None, days=30, output_format='pdf'):
        """
        Generate a comprehensive procurement analysis report
        
        Args:
            data_file (str): Optional path to data file (if not provided, will use recent data)
            days (int): Number of days of data to include if data_file not provided
            output_format (str): Output format ('pdf', 'html', or 'json')
            
        Returns:
            str: Path to the generated report
        """
        logger.info(f"Generating report (days={days}, format={output_format})")
        
        try:
            # Step 1: Get data
            if data_file and os.path.exists(data_file):
                print(f"Using provided data file: {data_file}")
                df = pd.read_csv(data_file)
            else:
                print(f"Retrieving data from the last {days} days...")
                
                # Get data from storage
                filters = {
                    'start_date': (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                    'end_date': datetime.now().strftime("%Y-%m-%d")
                }
                
                df = self.storage.retrieve_data(filters=filters, limit=10000)
                
                if df.empty:
                    print("No data available for report")
                    return None
            
            # Step 2: Run outlier detection if needed
            if 'is_outlier' not in df.columns:
                print("Running outlier detection...")
                
                # Load model
                if not self.model:
                    model_path = os.path.join(self.directories["models"], "isolation_forest_model.pkl")
                    
                    if os.path.exists(model_path):
                        self.model = IsolationForestModel(model_path=model_path)
                        self.model.load_model()
                    else:
                        print("No trained model found. Training new model...")
                        self.train(input_file=self.training_data_file)
                
                # Make predictions
                df = self.model.predict(df)
            
            # Step 3: Generate visualizations
            print("Generating visualizations...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            viz_files = self.visualizer.create_visualizations(
                df, 
                base_filename=f"report_{timestamp}"
            )
            
            # Step 4: Generate report based on format
            report_dir = os.path.join(self.base_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            report_file = os.path.join(report_dir, f"procurement_report_{timestamp}.{output_format}")
            
            # Generate different formats
            if output_format == 'json':
                # Create JSON report
                report_data = {
                    "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_period": f"{days} days",
                    "record_count": len(df),
                    "outlier_count": int(df['is_outlier'].sum()) if 'is_outlier' in df.columns else 0,
                    "visualizations": viz_files,
                    "summary_statistics": self.storage.get_statistics()
                }
                
                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent=4)
                
            else:
                # For PDF/HTML, we'll just use text for now
                # In a real implementation, you would generate proper PDF/HTML
                with open(report_file, 'w') as f:
                    f.write(f"EU Procurement Analysis Report\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Data period: Last {days} days\n")
                    f.write(f"Records analyzed: {len(df)}\n")
                    
                    if 'is_outlier' in df.columns:
                        outlier_count = df['is_outlier'].sum()
                        outlier_pct = outlier_count / len(df) * 100
                        f.write(f"Outliers detected: {outlier_count} ({outlier_pct:.2f}%)\n")
                    
                    f.write(f"\nVisualizations:\n")
                    for viz in viz_files:
                        f.write(f"- {viz}\n")
            
            print(f"Report generated: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            print(f"Error: {str(e)}")
            return None

def main():
    """Main entry point for the command line interface"""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='EU Procurement Monitoring System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synchronize data from the last 30 days
  python ted_ml_pipeline.py sync --days 30
  
  # Train a model using default training data
  python ted_ml_pipeline.py train
  
  # Detect outliers in data from a specific date range
  python ted_ml_pipeline.py predict --start-date 20250101 --end-date 20250131 --visualize
  
  # Generate a report for the last 30 days
  python ted_ml_pipeline.py report --days 30
"""
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Synchronize data from TED API')
    sync_parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    sync_parser.add_argument('--country', type=str, help='Country code filter (ISO)')
    sync_parser.add_argument('--max-pages', type=int, default=5, help='Maximum number of pages to fetch')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train outlier detection model')
    train_parser.add_argument('--input', type=str, help='Path to training data file')
    train_parser.add_argument('--sample', type=int, help='Sample size for training')
    train_parser.add_argument('--contamination', type=float, default=0.05, 
                            help='Expected proportion of outliers (0.0-0.5)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Detect outliers in procurement data')
    predict_parser.add_argument('--input', type=str, help='Path to input data file')
    predict_parser.add_argument('--start-date', type=str, help='Start date (YYYYMMDD)')
    predict_parser.add_argument('--end-date', type=str, help='End date (YYYYMMDD)')
    predict_parser.add_argument('--country', type=str, help='Country code filter (ISO)')
    predict_parser.add_argument('--max-bid-amount', type=float, help='Maximum bid amount')
    predict_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    evaluate_parser.add_argument('--input', type=str, required=True, help='Path to evaluation data file')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate procurement analysis report')
    report_parser.add_argument('--input', type=str, help='Path to input data file')
    report_parser.add_argument('--days', type=int, default=30, help='Number of days to include in report')
    report_parser.add_argument('--format', type=str, choices=['pdf', 'html', 'json'], 
                             default='pdf', help='Report format')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = TEDMLPipeline()
    
    # Execute command
    if args.command == 'sync':
        pipeline.sync_data(
            days_back=args.days,
            country=args.country,
            max_pages=args.max_pages
        )
    elif args.command == 'train':
        pipeline.train(
            input_file=args.input,
            sample_size=args.sample,
            contamination=args.contamination
        )
    elif args.command == 'predict':
        pipeline.predict(
            input_file=args.input,
            country=args.country,
            start_date=args.start_date,
            end_date=args.end_date,
            max_bid_amount=args.max_bid_amount,
            visualize=args.visualize
        )
    elif args.command == 'evaluate':
        pipeline.evaluate(
            input_file=args.input
        )
    elif args.command == 'report':
        pipeline.generate_report(
            data_file=args.input,
            days=args.days,
            output_format=args.format
        )
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()