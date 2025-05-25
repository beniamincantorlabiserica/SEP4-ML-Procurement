#!/usr/bin/env python3
"""
Simplified TED Data Storage Module
A clean and simple database for storing model results and statistics.
"""
import os
import sqlite3
import pandas as pd
import json
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TEDDataStorage:
    """Simple storage for TED procurement analysis results"""
    
    def __init__(self, db_path="data/ted_results.db"):
        """Initialize storage with database path"""
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"Storage initialized: {self.db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table 1: Model runs metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    parameters TEXT,
                    execution_date TEXT NOT NULL,
                    record_count INTEGER,
                    outlier_count INTEGER,
                    outlier_percentage REAL,
                    notes TEXT
                )
            ''')
            
            # Table 2: All procurement results with outlier info
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS procurement_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    notice_id TEXT,
                    value_eur REAL,
                    country TEXT,
                    notice_type TEXT,
                    is_outlier INTEGER NOT NULL,
                    anomaly_score REAL,
                    FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_run_id ON procurement_results(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_outlier ON procurement_results(is_outlier)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_notice_id ON procurement_results(notice_id)')
            
            conn.commit()
    
    def store_results(self, model_type, result_df, parameters=None, notes=None):
        """
        Store complete model results in one operation
        
        Args:
            model_type (str): Type of model ('isolation_forest', 'dbscan', 'kmeans')
            result_df (pd.DataFrame): DataFrame with outlier detection results
            parameters (dict): Model parameters
            notes (str): Additional notes
            
        Returns:
            str: Run ID for the stored results
        """
        run_id = str(uuid.uuid4())
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate statistics
                total_records = len(result_df)
                outlier_count = int(result_df.get('is_outlier', pd.Series([False])).sum())
                outlier_percentage = (outlier_count / total_records * 100) if total_records > 0 else 0
                
                # Store run metadata
                cursor.execute('''
                    INSERT INTO model_runs 
                    (run_id, model_type, parameters, execution_date, record_count, 
                     outlier_count, outlier_percentage, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    model_type,
                    json.dumps(parameters) if parameters else None,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    total_records,
                    outlier_count,
                    outlier_percentage,
                    notes
                ))
                
                # Prepare procurement results data
                results_data = []
                for _, row in result_df.iterrows():
                    # Extract key fields with fallbacks
                    notice_id = self._extract_field(row, ['notice-identifier', 'notice_id', 'identifier'])
                    value_eur = self._extract_numeric_field(row, ['total-value-eur', 'value_eur', 'total-value'])
                    country = self._extract_field(row, ['organisation-country-buyer', 'country'])
                    notice_type = self._extract_field(row, ['notice-type', 'notice_type'])
                    is_outlier = int(row.get('is_outlier', False))
                    anomaly_score = float(row.get('anomaly_score', 0) or 0)
                    
                    results_data.append((
                        run_id, notice_id, value_eur, country, notice_type, 
                        is_outlier, anomaly_score
                    ))
                
                # Store all procurement results
                cursor.executemany('''
                    INSERT INTO procurement_results 
                    (run_id, notice_id, value_eur, country, notice_type, is_outlier, anomaly_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', results_data)
                
                conn.commit()
                
            logger.info(f"Stored results: {run_id} ({model_type}) - {total_records} records, {outlier_count} outliers ({outlier_percentage:.2f}%)")
            return run_id
            
        except Exception as e:
            logger.error(f"Error storing results: {e}")
            return None
    
    def _extract_field(self, row, possible_names, default='unknown'):
        """Extract field value trying multiple possible column names"""
        for name in possible_names:
            if name in row.index and pd.notna(row[name]):
                return str(row[name])
        return default
    
    def _extract_numeric_field(self, row, possible_names, default=0.0):
        """Extract numeric field value trying multiple possible column names"""
        for name in possible_names:
            if name in row.index and pd.notna(row[name]):
                try:
                    return float(row[name])
                except (ValueError, TypeError):
                    continue
        return default
    
    def get_run_summary(self, run_id):
        """Get summary statistics for a specific run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get run metadata
            cursor.execute('''
                SELECT * FROM model_runs WHERE run_id = ?
            ''', (run_id,))
            
            run_info = cursor.fetchone()
            if not run_info:
                return None
            
            # Convert to dict and parse parameters
            result = dict(run_info)
            if result['parameters']:
                try:
                    result['parameters'] = json.loads(result['parameters'])
                except:
                    pass
            
            # Get outlier details
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_outliers,
                    AVG(anomaly_score) as avg_score,
                    MIN(value_eur) as min_value,
                    MAX(value_eur) as max_value,
                    AVG(value_eur) as avg_value
                FROM procurement_results 
                WHERE run_id = ? AND is_outlier = 1
            ''', (run_id,))
            
            outlier_stats = dict(cursor.fetchone())
            result['outlier_stats'] = outlier_stats
            
            # Get country breakdown
            cursor.execute('''
                SELECT 
                    country,
                    COUNT(*) as total,
                    SUM(is_outlier) as outliers,
                    ROUND(AVG(is_outlier) * 100, 2) as outlier_pct
                FROM procurement_results 
                WHERE run_id = ?
                GROUP BY country
                ORDER BY total DESC
                LIMIT 10
            ''', (run_id,))
            
            result['country_breakdown'] = [dict(row) for row in cursor.fetchall()]
            
            return result
    
    def get_recent_runs(self, limit=10):
        """Get recent model runs with basic statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    run_id,
                    model_type,
                    execution_date,
                    record_count,
                    outlier_count,
                    outlier_percentage,
                    notes
                FROM model_runs 
                ORDER BY execution_date DESC 
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_outliers(self, run_id, limit=100):
        """Get outliers from a specific run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT *
                FROM procurement_results 
                WHERE run_id = ? AND is_outlier = 1
                ORDER BY anomaly_score DESC
                LIMIT ?
            ''', (run_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_run_data(self, run_id):
        """Get all data from a specific run as DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM procurement_results WHERE run_id = ?
            '''
            return pd.read_sql_query(query, conn, params=(run_id,))
    
    def export_run_csv(self, run_id, output_file):
        """Export run data to CSV"""
        try:
            df = self.get_run_data(run_id)
            if df.empty:
                logger.warning(f"No data found for run {run_id}")
                return False
            
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(df)} records to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def get_statistics(self):
        """Get overall database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) as total_runs FROM model_runs')
            total_runs = cursor.fetchone()['total_runs']
            
            cursor.execute('SELECT COUNT(*) as total_records FROM procurement_results')
            total_records = cursor.fetchone()['total_records']
            
            cursor.execute('SELECT COUNT(*) as total_outliers FROM procurement_results WHERE is_outlier = 1')
            total_outliers = cursor.fetchone()['total_outliers']
            
            # Date range
            cursor.execute('SELECT MIN(execution_date) as first_run, MAX(execution_date) as last_run FROM model_runs')
            date_range = dict(cursor.fetchone())
            
            # Model type distribution
            cursor.execute('''
                SELECT model_type, COUNT(*) as count 
                FROM model_runs 
                GROUP BY model_type 
                ORDER BY count DESC
            ''')
            model_types = {row['model_type']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total_runs': total_runs,
                'total_records': total_records,
                'total_outliers': total_outliers,
                'overall_outlier_percentage': (total_outliers / total_records * 100) if total_records > 0 else 0,
                'date_range': date_range,
                'model_types': model_types
            }
    
    def cleanup_old_runs(self, days_to_keep=30):
        """Remove runs older than specified days"""
        from datetime import datetime, timedelta
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get runs to delete
            cursor.execute('''
                SELECT run_id FROM model_runs WHERE execution_date < ?
            ''', (cutoff_date,))
            
            old_runs = [row[0] for row in cursor.fetchall()]
            
            if old_runs:
                # Delete procurement results first (foreign key constraint)
                placeholders = ','.join(['?'] * len(old_runs))
                cursor.execute(f'''
                    DELETE FROM procurement_results WHERE run_id IN ({placeholders})
                ''', old_runs)
                
                # Delete model runs
                cursor.execute(f'''
                    DELETE FROM model_runs WHERE run_id IN ({placeholders})
                ''', old_runs)
                
                conn.commit()
                logger.info(f"Cleaned up {len(old_runs)} old runs")
                
            return len(old_runs)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TED Data Storage Management')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--runs', action='store_true', help='Show recent runs')
    parser.add_argument('--run-details', type=str, help='Show details for specific run ID')
    parser.add_argument('--outliers', type=str, help='Show outliers for specific run ID')
    parser.add_argument('--export', nargs=2, help='Export run data: run_id output_file')
    parser.add_argument('--cleanup', type=int, help='Clean up runs older than X days')
    
    args = parser.parse_args()
    
    # Initialize storage
    storage = TEDDataStorage()
    
    if args.stats:
        stats = storage.get_statistics()
        print("\n=== DATABASE STATISTICS ===")
        print(f"Total Runs: {stats['total_runs']}")
        print(f"Total Records: {stats['total_records']:,}")
        print(f"Total Outliers: {stats['total_outliers']:,} ({stats['overall_outlier_percentage']:.2f}%)")
        print(f"Date Range: {stats['date_range']['first_run']} to {stats['date_range']['last_run']}")
        print(f"Model Types: {stats['model_types']}")
    
    elif args.runs:
        runs = storage.get_recent_runs(20)
        print("\n=== RECENT RUNS ===")
        for run in runs:
            print(f"{run['execution_date']} | {run['model_type']} | {run['record_count']:,} records | {run['outlier_count']:,} outliers ({run['outlier_percentage']:.2f}%)")
            print(f"  Run ID: {run['run_id']}")
            if run['notes']:
                print(f"  Notes: {run['notes']}")
            print()
    
    elif args.run_details:
        details = storage.get_run_summary(args.run_details)
        if details:
            print(f"\n=== RUN DETAILS: {args.run_details} ===")
            print(f"Model Type: {details['model_type']}")
            print(f"Date: {details['execution_date']}")
            print(f"Records: {details['record_count']:,}")
            print(f"Outliers: {details['outlier_count']:,} ({details['outlier_percentage']:.2f}%)")
            
            if details['outlier_stats']['total_outliers']:
                print(f"\nOutlier Statistics:")
                print(f"  Average Score: {details['outlier_stats']['avg_score']:.4f}")
                print(f"  Value Range: €{details['outlier_stats']['min_value']:,.2f} - €{details['outlier_stats']['max_value']:,.2f}")
                print(f"  Average Value: €{details['outlier_stats']['avg_value']:,.2f}")
            
            if details['country_breakdown']:
                print(f"\nTop Countries:")
                for country in details['country_breakdown'][:5]:
                    print(f"  {country['country']}: {country['total']:,} records, {country['outliers']} outliers ({country['outlier_pct']}%)")
        else:
            print(f"Run ID not found: {args.run_details}")
    
    elif args.outliers:
        outliers = storage.get_outliers(args.outliers, 20)
        if outliers:
            print(f"\n=== OUTLIERS FOR RUN: {args.outliers} ===")
            for outlier in outliers:
                print(f"Notice: {outlier['notice_id']} | Value: €{outlier['value_eur']:,.2f} | Score: {outlier['anomaly_score']:.4f} | Country: {outlier['country']}")
        else:
            print(f"No outliers found for run: {args.outliers}")
    
    elif args.export:
        run_id, output_file = args.export
        success = storage.export_run_csv(run_id, output_file)
        if success:
            print(f"Data exported to: {output_file}")
        else:
            print("Export failed")
    
    elif args.cleanup:
        deleted = storage.cleanup_old_runs(args.cleanup)
        print(f"Cleaned up {deleted} runs older than {args.cleanup} days")
    
    else:
        parser.print_help()