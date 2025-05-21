#!/usr/bin/env python3
"""
TED Data Storage Module
This module handles the persistent storage of processed procurement data.
It provides functionality for storing, retrieving, and managing datasets.

"""
import os
import pandas as pd
import shutil
from datetime import datetime
import json
import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TEDDataStorage:
    """Class for persistent storage of processed TED procurement data"""
    
    def __init__(self, data_dir="data/processed", db_path=None, max_storage_days=365):
        """
        Initialize the data storage component
        
        Args:
            data_dir (str): Directory for CSV storage
            db_path (str): Path to SQLite database file (if None, defaults to data_dir/ted_data.db)
            max_storage_days (int): Maximum number of days to keep historical data
        """
        self.data_dir = data_dir
        self.db_path = db_path or os.path.join(data_dir, "ted_data.db")
        self.max_storage_days = max_storage_days
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize metadata if needed
        if not os.path.exists(self.metadata_file):
            self._init_metadata()
            
    def _init_database(self):
        """Initialize the SQLite database with required schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main table for storing procurement data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS procurement_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notice_identifier TEXT,
                value_eur REAL,
                country TEXT,
                notice_type TEXT,
                created_date TEXT,
                bidder_count INTEGER,
                bidder_size TEXT,
                data_file TEXT,
                import_date TEXT
            )
            ''')
            
            # Create table for outliers
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS outliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notice_identifier TEXT,
                value_eur REAL,
                outlier_score REAL,
                detection_date TEXT,
                model_version TEXT
            )
            ''')
            
            # Create index on notice_identifier
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_notice_id 
            ON procurement_data (notice_identifier)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            
    def _init_metadata(self):
        """Initialize the metadata file"""
        metadata = {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": 0,
            "data_files": [],
            "models": []
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info("Metadata file initialized")
        
    def _update_metadata(self, new_file=None, new_model=None):
        """Update the metadata file with new information"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Update timestamp
            metadata["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update record count
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM procurement_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            metadata["total_records"] = count
            
            # Add new file if provided
            if new_file and new_file not in metadata["data_files"]:
                metadata["data_files"].append(new_file)
                
            # Add new model if provided
            if new_model and new_model not in metadata["models"]:
                metadata["models"].append(new_model)
                
            # Write updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info("Metadata updated successfully")
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            
    def store_data(self, file_path):
        """
        Store processed data in the persistent storage
        
        Args:
            file_path (str): Path to the CSV file to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            
            # Make a copy in the data directory
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.csv"
            new_path = os.path.join(self.data_dir, new_filename)
            
            shutil.copy2(file_path, new_path)
            logger.info(f"Copied file to {new_path}")
            
            # Extract key data for database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data for insertion
            records = []
            for _, row in df.iterrows():
                # Extract relevant fields if available
                notice_id = row.get('notice-identifier', row.get('identifier', 'unknown'))
                value_eur = row.get('total-value-eur', row.get('value', 0))
                country = row.get('organisation-country-buyer', row.get('country', 'unknown'))
                notice_type = row.get('notice-type', 'unknown')
                bidder_count = row.get('bidder-count', 0)
                bidder_size = row.get('primary-bidder-size', 'unknown')
                
                record = (
                    notice_id,
                    value_eur,
                    country,
                    notice_type,
                    row.get('publication-date', datetime.now().strftime("%Y-%m-%d")),
                    bidder_count,
                    bidder_size,
                    new_filename,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                records.append(record)
            
            # Insert records into database
            cursor.executemany('''
            INSERT INTO procurement_data 
            (notice_identifier, value_eur, country, notice_type, created_date, 
             bidder_count, bidder_size, data_file, import_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            conn.close()
            
            # Update metadata
            self._update_metadata(new_file=new_filename)
            
            logger.info(f"Successfully stored {len(records)} records in the database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return False
            
    def retrieve_data(self, filters=None, limit=1000):
        """
        Retrieve data from storage based on filters
        
        Args:
            filters (dict): Optional filters to apply
            limit (int): Maximum number of records to retrieve
            
        Returns:
            pd.DataFrame: Retrieved data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query based on filters
            query = "SELECT * FROM procurement_data"
            params = []
            
            if filters:
                conditions = []
                
                if 'country' in filters:
                    conditions.append("country = ?")
                    params.append(filters['country'])
                    
                if 'min_value' in filters:
                    conditions.append("value_eur >= ?")
                    params.append(filters['min_value'])
                    
                if 'max_value' in filters:
                    conditions.append("value_eur <= ?")
                    params.append(filters['max_value'])
                    
                if 'notice_type' in filters:
                    conditions.append("notice_type = ?")
                    params.append(filters['notice_type'])
                    
                if 'start_date' in filters:
                    conditions.append("created_date >= ?")
                    params.append(filters['start_date'])
                    
                if 'end_date' in filters:
                    conditions.append("created_date <= ?")
                    params.append(filters['end_date'])
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            # Add limit
            query += f" LIMIT {limit}"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            logger.info(f"Retrieved {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return pd.DataFrame()
            
    def store_outliers(self, outliers_df, model_version):
        """
        Store detected outliers in the database
        
        Args:
            outliers_df (pd.DataFrame): DataFrame containing outliers
            model_version (str): Version or identifier of the model used
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Filter to only outliers
            if 'is_outlier' in outliers_df.columns:
                outliers_only = outliers_df[outliers_df['is_outlier'] == True].copy()
            else:
                outliers_only = outliers_df.copy()
                
            if len(outliers_only) == 0:
                logger.info("No outliers to store")
                return True
                
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare records for insertion
            detection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            records = []
            
            for _, row in outliers_only.iterrows():
                # Extract relevant fields
                notice_id = row.get('notice-identifier', row.get('identifier', 'unknown'))
                value_eur = row.get('total-value-eur', row.get('value', 0))
                score = row.get('anomaly_score', 0)
                
                record = (
                    notice_id,
                    value_eur,
                    score,
                    detection_date,
                    model_version
                )
                records.append(record)
                
            # Insert records
            cursor.executemany('''
            INSERT INTO outliers
            (notice_identifier, value_eur, outlier_score, detection_date, model_version)
            VALUES (?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(records)} outliers in the database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing outliers: {e}")
            return False
            
    def cleanup_old_data(self):
        """
        Remove data older than max_storage_days
        
        Returns:
            int: Number of records removed
        """
        try:
            # Calculate cutoff date
            cutoff_date = (datetime.now() - pd.Timedelta(days=self.max_storage_days)).strftime("%Y-%m-%d")
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count records to remove
            cursor.execute(
                "SELECT COUNT(*) FROM procurement_data WHERE created_date < ?",
                (cutoff_date,)
            )
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Remove old records
                cursor.execute(
                    "DELETE FROM procurement_data WHERE created_date < ?",
                    (cutoff_date,)
                )
                
                conn.commit()
                logger.info(f"Removed {count} old records from database")
                
                # Update metadata
                self._update_metadata()
                
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
            
    def export_data(self, output_file, filters=None):
        """
        Export data from database to CSV file
        
        Args:
            output_file (str): Path to output CSV file
            filters (dict): Optional filters to apply
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Retrieve data with filters
            df = self.retrieve_data(filters=filters, limit=100000)
            
            if df.empty:
                logger.warning("No data to export")
                return False
                
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} records to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
            
    def get_statistics(self):
        """
        Get statistics about the stored data
        
        Returns:
            dict: Statistics about the data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total record count
            cursor.execute("SELECT COUNT(*) FROM procurement_data")
            total_count = cursor.fetchone()[0]
            
            # Get outlier count
            cursor.execute("SELECT COUNT(*) FROM outliers")
            outlier_count = cursor.fetchone()[0]
            
            # Get count by country
            cursor.execute(
                "SELECT country, COUNT(*) FROM procurement_data GROUP BY country ORDER BY COUNT(*) DESC LIMIT 10"
            )
            countries = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get average value
            cursor.execute("SELECT AVG(value_eur) FROM procurement_data")
            avg_value = cursor.fetchone()[0]
            
            # Get most recent data
            cursor.execute(
                "SELECT MAX(import_date) FROM procurement_data"
            )
            latest_data = cursor.fetchone()[0]
            
            # Get value distribution
            cursor.execute(
                """
                SELECT 
                    CASE 
                        WHEN value_eur < 10000 THEN 'Under €10K'
                        WHEN value_eur < 100000 THEN '€10K-€100K'
                        WHEN value_eur < 1000000 THEN '€100K-€1M'
                        WHEN value_eur < 10000000 THEN '€1M-€10M'
                        ELSE 'Over €10M'
                    END as value_range,
                    COUNT(*) as count
                FROM procurement_data
                GROUP BY value_range
                ORDER BY MIN(value_eur)
                """
            )
            value_dist = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            # Compile statistics
            stats = {
                "total_records": total_count,
                "total_outliers": outlier_count,
                "outlier_percentage": round(outlier_count / total_count * 100, 2) if total_count > 0 else 0,
                "country_distribution": countries,
                "average_value_eur": round(avg_value, 2) if avg_value else 0,
                "latest_data_import": latest_data,
                "value_distribution": value_dist
            }
            
            logger.info("Retrieved database statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            return {
                "error": str(e),
                "total_records": 0,
                "total_outliers": 0
            }
            
# If run directly, perform a self-test
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='TED Data Storage Management')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old data')
    parser.add_argument('--export', type=str, help='Export data to CSV file')
    parser.add_argument('--days', type=int, default=365, help='Maximum age of data in days')
    
    args = parser.parse_args()
    
    # Create storage instance
    storage = TEDDataStorage(max_storage_days=args.days)
    
    if args.stats:
        stats = storage.get_statistics()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
                
    if args.cleanup:
        removed = storage.cleanup_old_data()
        print(f"\nRemoved {removed} records older than {args.days} days")
        
    if args.export:
        success = storage.export_data(args.export)
        if success:
            print(f"\nData exported successfully to {args.export}")
        else:
            print("\nFailed to export data")