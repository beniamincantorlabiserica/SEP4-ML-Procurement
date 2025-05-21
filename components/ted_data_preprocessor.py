#!/usr/bin/env python3
"""
TED Data Preprocessor Module

This module handles the preprocessing of TED procurement data for machine learning.
It cleans, normalizes, and transforms the data to make it suitable for outlier detection.

Author: Your Name
Date: May 16, 2025
"""

import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime

class TEDDataPreprocessor:
    """Class for preprocessing TED procurement data"""
    
    def __init__(self, input_file=None, output_dir="processed_data"):
        self.input_file = input_file
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def load_csv_safely(self, file_path):
        """
        Load a CSV file with robust error handling for inconsistent field counts
        """
        print(f"Loading file: {file_path}")
        try:
            # First try pandas with default settings
            df = pd.read_csv(file_path)
            print(f"Successfully loaded with pandas: {len(df)} rows")
            return df
        except Exception as e:
            print(f"Standard loading failed: {str(e)}")
            print("Trying alternative loading method...")
            
            # Manual loading using csv module
            rows = []
            header = None
            max_fields = 0
            
            # First pass to get header and max field count
            with open(file_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        header = row
                    else:
                        max_fields = max(max_fields, len(row))
                
            max_fields = max(max_fields, len(header))
            print(f"Max field count: {max_fields}")
            
            # Second pass to read the data
            with open(file_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    # Pad or truncate row
                    if len(row) < max_fields:
                        row = row + [''] * (max_fields - len(row))
                    elif len(row) > max_fields:
                        row = row[:max_fields]
                    rows.append(row)
            
            # Ensure header has the right length
            if len(header) < max_fields:
                header.extend([f"unknown_{i}" for i in range(len(header), max_fields)])
            elif len(header) > max_fields:
                header = header[:max_fields]
                
            # Create DataFrame
            df = pd.DataFrame(rows, columns=header)
            print(f"Successfully loaded with manual method: {len(df)} rows, {len(df.columns)} columns")
            return df
    
    def load_data(self):
        """Load the TED procurement data"""
        if not self.input_file:
            raise ValueError("Input file not specified")
        
        return self.load_csv_safely(self.input_file)
    
    def clean_data_for_ml(self, df):
        """
        Clean and normalize TED procurement data for machine learning
        Parameters:
        -----------
        df : pd.DataFrame
            Raw TED procurement data
        Returns:
        --------
        pd.DataFrame
            Cleaned data ready for ML
        """
        print("\nCleaning and normalizing data...")
        cleaned_df = df.copy()
        
        # Step 1: Remove link fields
        link_cols = [col for col in cleaned_df.columns if 
                    'link' in col.lower() or 
                    'url' in col.lower() or 
                    'xml' in col.lower() or 
                    'html' in col.lower() or 
                    'pdf' in col.lower()]
        
        if link_cols:
            print(f"Removing {len(link_cols)} link-related columns")
            cleaned_df = cleaned_df.drop(columns=link_cols)
        
        # Step 2: Extract clean currency information
        if 'estimated-value-cur-proc' in cleaned_df.columns:
            valid_currencies = ['EUR', 'SEK', 'BGN', 'NOK', 'PLN', 'CZK', 'HUF', 'DKK', 'RON']
            
            def extract_currency(value):
                if pd.isna(value) or not isinstance(value, str):
                    return 'EUR'  # Default currency
                for curr in valid_currencies:
                    if curr in value:
                        return curr
                return 'EUR'
            
            cleaned_df['currency'] = cleaned_df['estimated-value-cur-proc'].apply(extract_currency)
            currency_counts = cleaned_df['currency'].value_counts()
            print(f"Currency distribution: {dict(currency_counts)}")
        else:
            # Default currency if not present
            cleaned_df['currency'] = 'EUR'
        
        # Step 3: Process monetary values
        for col in ['total-value', 'framework-value-notice', 'subcontracting-value']:
            if col in cleaned_df.columns:
                # Convert to string first
                cleaned_df[col] = cleaned_df[col].astype(str)
                # Clean up the values
                cleaned_df[col] = cleaned_df[col].str.replace(',', '.')
                cleaned_df[col] = cleaned_df[col].str.replace(r'[^\d.]', '', regex=True)
                # Convert to numeric
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                valid_count = cleaned_df[col].count()
                print(f"Processed {col}: {valid_count} valid values")
        
        # Step 4: Normalize monetary values to EUR
        exchange_rates = {
            'EUR': 1.0,
            'SEK': 0.087,
            'BGN': 0.51,
            'NOK': 0.086,
            'PLN': 0.23,
            'CZK': 0.039,
            'HUF': 0.0026,
            'DKK': 0.13,
            'RON': 0.20
        }
        
        if 'total-value' in cleaned_df.columns:
            cleaned_df['total-value-eur'] = cleaned_df.apply(
                lambda row: row['total-value'] * exchange_rates.get(row['currency'], 1.0) 
                if pd.notna(row['total-value']) else np.nan,
                axis=1
            )
            print(f"Normalized total values to EUR: {cleaned_df['total-value-eur'].count()} values")
            
            # Cap outliers for better model stability
            # Calculate 95th percentile for capping
            percentile_95 = cleaned_df['total-value-eur'].quantile(0.95)
            cleaned_df['total-value-eur-capped'] = cleaned_df['total-value-eur'].apply(
                lambda x: min(x, percentile_95) if pd.notna(x) else x
            )
            
            # Add outlier flag based on simple threshold for initial filtering
            if 'total-value-eur' in cleaned_df.columns:
                # Flag values above 95th percentile as potential outliers
                cleaned_df['is_outlier'] = (cleaned_df['total-value-eur'] > percentile_95).astype(bool)
            
            # Log transform of monetary values (useful for ML)
            cleaned_df['total-value-eur-log'] = np.log1p(
                cleaned_df['total-value-eur'].replace([np.inf, -np.inf, np.nan], 0)
            )
        
        # Step 5: Extract bidder information
        if 'winner-size' in cleaned_df.columns:
            try:
                # Count bidders
                cleaned_df['bidder-count'] = cleaned_df['winner-size'].astype(str).apply(
                    lambda x: len(x.split('|')) if pd.notna(x) and x != 'nan' and x != 'None' else 0
                )
                
                # Extract primary bidder size
                cleaned_df['primary-bidder-size'] = cleaned_df['winner-size'].astype(str).apply(
                    lambda x: x.split('|')[0] if pd.notna(x) and x != 'nan' and x != 'None' else np.nan
                )
                
                # Convert size to numeric representation
                size_mapping = {
                    'micro': 1,
                    'small': 2,
                    'sme': 2.5,  # between small and medium
                    'medium': 3,
                    'large': 4
                }
                cleaned_df['bidder-size-numeric'] = cleaned_df['primary-bidder-size'].map(size_mapping)
                
                print(f"Extracted bidder info: max bidders = {cleaned_df['bidder-count'].max()}")
            except Exception as e:
                print(f"Error extracting bidder information: {e}")
        
        # Step 6: Format categorical features
        if 'notice-type' in cleaned_df.columns:
            # Get top notice types
            top_types = cleaned_df['notice-type'].value_counts().head(10).index.tolist()
            
            # Create dummy variables for top types
            for notice_type in top_types:
                col_name = f"notice_is_{notice_type}"
                cleaned_df[col_name] = (cleaned_df['notice-type'] == notice_type).astype(int)
            
            print(f"Created dummy variables for {len(top_types)} notice types")
        
        return cleaned_df
    
    def prepare_for_ml(self, df):
        """
        Final preparation to make the data compatible with the ML algorithm
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned DataFrame
        Returns:
        --------
        pd.DataFrame
            ML-ready DataFrame with only relevant features
        """
        print("\nPreparing final ML-ready dataset...")
        
        # Focus on rows with valid monetary values
        if 'total-value-eur' in df.columns:
            ml_df = df.dropna(subset=['total-value-eur']).copy()
            print(f"Kept {len(ml_df)}/{len(df)} rows with valid monetary values")
        else:
            ml_df = df.copy()
            print("Warning: No monetary values found")
        
        # Select features important for ML
        keep_columns = []
        
        # Always include ID if available
        id_cols = [col for col in ml_df.columns if 'identifier' in col.lower() or 'id' in col.lower()]
        if id_cols:
            keep_columns.extend(id_cols[:1])  # Take the first ID column
        
        # Include monetary values
        money_cols = ['total-value-eur', 'total-value-eur-capped', 'total-value-eur-log']
        keep_columns.extend([col for col in money_cols if col in ml_df.columns])
        
        # Include bidder info
        bidder_cols = ['bidder-count', 'bidder-size-numeric']
        keep_columns.extend([col for col in bidder_cols if col in ml_df.columns])
        
        # Include notice type dummies
        notice_dummies = [col for col in ml_df.columns if col.startswith('notice_is_')]
        keep_columns.extend(notice_dummies)
        
        # Filter to only existing columns
        keep_columns = [col for col in keep_columns if col in ml_df.columns]
        
        # Keep other potentially useful columns
        remain_cols = []
        for col in ml_df.columns:
            # Skip already included columns
            if col in keep_columns:
                continue
                
            # Skip text fields and other less useful columns
            if ('text' in col.lower() or 
                'description' in col.lower() or 
                'currency' in col.lower() or 
                'link' in col.lower()):
                continue
                
            # Keep numeric columns with reasonable non-null counts
            if ml_df[col].dtype in ['int64', 'float64']:
                non_null_pct = ml_df[col].count() / len(ml_df)
                if non_null_pct > 0.5:  # At least 50% non-null
                    remain_cols.append(col)
        
        # Add remaining useful columns
        keep_columns.extend(remain_cols[:5])  # Limit to 5 additional columns
        
        # Create final dataset
        final_df = ml_df[keep_columns].copy()
        print(f"Final ML dataset: {len(final_df)} rows, {len(keep_columns)} columns")
        print(f"Features included: {keep_columns}")
        
        return final_df
    
    def preprocess_data(self):
        """
        Run the full preprocessing pipeline
        Returns:
        --------
        tuple
            (normalized_data, ml_ready_data)
        """
        # Load data
        df = self.load_data()
        
        # Clean and normalize
        normalized_df = self.clean_data_for_ml(df)
        
        # Prepare for ML
        ml_df = self.prepare_for_ml(normalized_df)
        
        return normalized_df, ml_df
    
    def save_output(self):
        """
        Save the preprocessed data to CSV files
        Returns:
        --------
        dict
            Paths to the saved files
        """
        # Run preprocessing
        normalized_df, ml_df = self.preprocess_data()
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save normalized data
        normalized_path = os.path.join(self.output_dir, f"ted_normalized_{timestamp}.csv")
        normalized_df.to_csv(normalized_path, index=False)
        print(f"Saved normalized data to: {normalized_path}")
        
        # Save ML-ready data
        ml_path = os.path.join(self.output_dir, f"ted_ml_dataset_{timestamp}.csv")
        ml_df.to_csv(ml_path, index=False)
        print(f"Saved ML-ready data to: {ml_path}")
        
        return {
            "normalized": normalized_path,
            "ml_dataset": ml_path,
            "ml_df": ml_df
        }


# If run directly, perform a test preprocessing
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preprocess TED procurement data for ML')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='processed_data', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = TEDDataPreprocessor(args.input, args.output)
    result = preprocessor.save_output()
    
    print("\nPreprocessing summary:")
    print(f"Input file: {args.input}")
    print(f"Normalized data saved to: {result['normalized']}")
    print(f"ML-ready data saved to: {result['ml_dataset']}")
    print(f"ML dataset shape: {result['ml_df'].shape}")