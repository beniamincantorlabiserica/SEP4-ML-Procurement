#!/usr/bin/env python3
"""
Clustering Model Module for TED Procurement Outlier Detection
This module provides an alternative approach to outlier detection using clustering techniques.
It handles training, prediction, model serialization, and evaluation.

Author: Your Name
Date: May 21, 2025
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# Set random seed for reproducibility
np.random.seed(42)

class ClusteringModel:
    """Class for building and using clustering models for outlier detection"""
    
    def __init__(self, model_path=None, eps=0.5, min_samples=5, model_dir="models", viz_dir="visualizations"):
        """
        Initialize the clustering model
        
        Args:
            model_path (str): Path to save/load model
            eps (float): DBSCAN parameter - maximum distance between samples
            min_samples (int): DBSCAN parameter - minimum samples in neighborhood
            model_dir (str): Directory for model storage
            viz_dir (str): Directory for visualizations
        """
        self.model_dir = model_dir
        self.viz_dir = viz_dir
        self.model_path = model_path or os.path.join(model_dir, "clustering_model.pkl")
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.feature_columns = None
        self.numerical_features = None
        self.categorical_features = None
        self.scaler = None
        
        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
    def prepare_features(self, df):
        """
        Prepare features for the clustering model
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        print("\nPreparing features for clustering...")
        
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
        Build a preprocessing pipeline for clustering
        
        Args:
            numerical_features (list): List of numerical feature names
            categorical_features (list): List of categorical feature names
            
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        print("\nBuilding preprocessing pipeline...")
        
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
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def find_optimal_eps(self, X, n_samples=1000):
        """
        Find optimal epsilon parameter for DBSCAN
        
        Args:
            X (np.ndarray): Preprocessed feature array
            n_samples (int): Number of samples to use for estimation
            
        Returns:
            float: Estimated optimal epsilon value
        """
        # Sample data if needed
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        # Compute nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(X_sample)
        distances, _ = nbrs.kneighbors(X_sample)
        
        # Sort and get distance to kth neighbor
        distances = np.sort(distances[:, self.min_samples-1])
        
        # Estimate optimal epsilon using the "elbow" method
        # Calculate the rate of change in distances
        diffs = np.diff(distances)
        
        # Find the point where the rate of change is greatest
        elbow_idx = np.argmax(diffs) + 1
        optimal_eps = distances[elbow_idx]
        
        print(f"Estimated optimal epsilon: {optimal_eps:.4f}")
        return optimal_eps
    
    def train(self, df, sample_size=None, use_kmeans=False, n_clusters=5):
        """
        Train a clustering model on the dataset
        
        Args:
            df (pd.DataFrame): Input DataFrame for training
            sample_size (int): Optional sample size to use
            use_kmeans (bool): Whether to use KMeans instead of DBSCAN
            n_clusters (int): Number of clusters for KMeans
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("\nTraining clustering model...")
        
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
        X_df = df_sample[numerical_features + categorical_features].copy()
        
        try:
            # Build preprocessing pipeline
            preprocessor = self.build_pipeline(numerical_features, categorical_features)
            
            # Fit and transform the data
            X = preprocessor.fit_transform(X_df)
            print(f"Preprocessed data shape: {X.shape}")
            
            # Choose clustering algorithm
            if use_kmeans:
                print(f"Training KMeans with {n_clusters} clusters...")
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            else:
                # Find optimal epsilon if not manually set
                if self.eps is None or self.eps <= 0:
                    self.eps = self.find_optimal_eps(X)
                    
                print(f"Training DBSCAN with eps={self.eps}, min_samples={self.min_samples}...")
                model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
            
            # Fit the model
            model.fit(X)
            
            # Store model and features
            self.model = model
            self.feature_columns = X_df.columns.tolist()
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features
            self.preprocessor = preprocessor
            
            # Get cluster labels
            if use_kmeans:
                labels = model.labels_
                n_clusters = len(set(labels))
                print(f"Model trained with {n_clusters} clusters")
            else:
                labels = model.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                print(f"Model trained with {n_clusters} clusters and {n_outliers} outliers")
                print(f"Outlier percentage: {n_outliers/len(labels)*100:.2f}%")
            
            print("Model training completed successfully.")
            return True
            
        except Exception as e:
            print(f"Error during model training: {e}")
            
            # Try with only numerical features if there was an error
            print("Attempting to train with only numerical features...")
            try:
                preprocessor = self.build_pipeline(numerical_features, [])
                X_num = df_sample[numerical_features].copy()
                X = preprocessor.fit_transform(X_num)
                
                if use_kmeans:
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                else:
                    # Find optimal epsilon if not manually set
                    if self.eps is None or self.eps <= 0:
                        self.eps = self.find_optimal_eps(X)
                        
                    model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
                
                model.fit(X)
                
                self.model = model
                self.feature_columns = numerical_features
                self.numerical_features = numerical_features
                self.categorical_features = []
                self.preprocessor = preprocessor
                
                print("Model training with numerical features only completed successfully.")
                return True
                
            except Exception as e2:
                print(f"Error during fallback training: {e2}")
                return False
    
    def predict(self, df):
        """
        Use the trained model to detect outliers in the dataset
        
        Args:
            df (pd.DataFrame): Dataset for prediction
            
        Returns:
            pd.DataFrame: DataFrame with added prediction results
        """
        print("\nDetecting outliers using clustering...")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
            
        try:
            # Ensure we have all required columns
            missing_columns = [col for col in self.feature_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in dataset: {missing_columns}")
                
                # Add missing columns with default values
                for col in missing_columns:
                    df[col] = 0
                print(f"Added missing columns with default values")
                
            # Prepare features for prediction
            all_feature_columns = [col for col in self.feature_columns if col in df.columns]
            X_df = df[all_feature_columns].copy()
            
            # Apply preprocessing
            X = self.preprocessor.transform(X_df)
            
            # Check if model is KMeans or DBSCAN
            if hasattr(self.model, 'predict'):
                # For KMeans, predict clusters and calculate distances to centroids
                clusters = self.model.predict(X)
                
                # Calculate distance to closest centroid
                distances = np.min(
                    np.sqrt(np.sum((X - self.model.cluster_centers_[clusters.reshape(-1, 1)])**2, axis=2)),
                    axis=1
                )
                
                # Define outliers as points with distance > threshold
                # Using 95th percentile as threshold
                threshold = np.percentile(distances, 95)
                outliers = distances > threshold
                
                # Calculate anomaly scores (normalized distances)
                scores = distances / np.max(distances)
                
            else:
                # For DBSCAN, use the model's labels
                # First, need to run fit_predict as DBSCAN doesn't have predict method
                clusters = self.model.fit_predict(X)
                
                # Points with cluster label -1 are outliers
                outliers = clusters == -1
                
                # Calculate anomaly scores based on distances to nearest core points
                # Note: This is a simplified approach
                nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(X)
                distances, _ = nbrs.kneighbors(X)
                
                # Use average distance to k nearest neighbors as score
                scores = np.mean(distances, axis=1)
                scores = scores / np.max(scores)  # Normalize
            
            # Add results to the dataframe
            result_df = df.copy()
            result_df['cluster'] = clusters
            result_df['is_outlier'] = outliers
            
            # Add clear text status
            result_df['outlier_status'] = result_df['is_outlier'].apply(
                lambda x: 'OUTLIER' if x else 'NORMAL'
            )
            
            # Add anomaly scores
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
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\nSaving model to {self.model_path}...")
        
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        
        # Create a package with all necessary components
        model_package = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'eps': self.eps,
            'min_samples': self.min_samples,
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
        Load a previously trained clustering model
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Loading model from {self.model_path}...")
        
        try:
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
                
            self.model = model_package['model']
            self.preprocessor = model_package['preprocessor']
            self.feature_columns = model_package['feature_columns']
            self.numerical_features = model_package['numerical_features']
            self.categorical_features = model_package['categorical_features']
            self.eps = model_package.get('eps', 0.5)
            self.min_samples = model_package.get('min_samples', 5)
            
            date_trained = model_package.get('date_trained', 'unknown')
            print(f"Model loaded successfully. Trained on: {date_trained}")
            
            print(f"Features: {len(self.feature_columns)} total features")
            print(f" - {len(self.numerical_features)} numerical features")
            print(f" - {len(self.categorical_features)} categorical features")
            
            # Check if model is KMeans or DBSCAN
            if hasattr(self.model, 'n_clusters'):
                print(f"KMeans model with {self.model.n_clusters} clusters")
            else:
                # For DBSCAN, count unique clusters
                if hasattr(self.model, 'labels_'):
                    labels = self.model.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_outliers = list(labels).count(-1)
                    print(f"DBSCAN model with {n_clusters} clusters")
                    print(f"Parameters: eps={self.eps}, min_samples={self.min_samples}")
                    print(f"Detected {n_outliers} outliers in training data ({n_outliers/len(labels)*100:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate(self, df):
        """
        Evaluate the clustering model performance
        
        Args:
            df (pd.DataFrame): Evaluation dataset
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print("\nEvaluating clustering model...")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
            
        try:
            # Prepare features for evaluation
            all_feature_columns = [col for col in self.feature_columns if col in df.columns]
            X_df = df[all_feature_columns].copy()
            
            # Apply preprocessing
            X = self.preprocessor.transform(X_df)
            
            # Check if model is KMeans or DBSCAN
            if hasattr(self.model, 'predict'):
                # For KMeans
                labels = self.model.predict(X)
                
                # Calculate silhouette score
                silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
                
                # Calculate inertia (sum of squared distances to centroids)
                inertia = self.model.inertia_
                
                metrics = {
                    'silhouette_score': silhouette,
                    'inertia': inertia,
                    'n_clusters': self.model.n_clusters
                }
                
            else:
                # For DBSCAN, refit on evaluation data
                labels = self.model.fit_predict(X)
                
                # Count clusters and outliers
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                outlier_percentage = n_outliers / len(labels) * 100
                
                # Calculate silhouette score for non-outlier points
                if n_clusters > 1:
                    # Filter out outliers for silhouette calculation
                    mask = labels != -1
                    if sum(mask) > 1 and len(set(labels[mask])) > 1:
                        silhouette = silhouette_score(X[mask], labels[mask])
                    else:
                        silhouette = 0
                else:
                    silhouette = 0
                
                metrics = {
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'outlier_percentage': outlier_percentage,
                    'silhouette_score': silhouette,
                    'eps': self.eps,
                    'min_samples': self.min_samples
                }
            
            print(f"Model evaluation completed:")
            for metric, value in metrics.items():
                print(f" - {metric}: {value}")
                
            return metrics
            
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return {'error': str(e)}
    
    def save_predictions(self, result_df, output_file=None):
        """
        Save the prediction results to a CSV file
        
        Args:
            result_df (pd.DataFrame): DataFrame with prediction results
            output_file (str): Path to save the CSV file
            
        Returns:
            str: Path to the saved file
        """
        # If no specific output file is provided, create one with timestamp
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join("results", f"clustering_outliers_{timestamp}.csv")
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"\nSaving predictions to {output_file}...")
        
        try:
            result_df.to_csv(output_file, index=False)
            print(f"Results saved successfully.")
            
            # Print summary
            outlier_count = result_df['is_outlier'].sum()
            total_count = len(result_df)
            print(f"Summary: {outlier_count} outliers detected out of {total_count} records "
                 f"({outlier_count/total_count*100:.2f}%)")
                 
            return output_file
            
        except Exception as e:
            print(f"Error saving predictions: {e}")
            return None

# If run directly, perform a test train and predict
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='TED Procurement Clustering for Outlier Detection')
    parser.add_argument('--input', type=str, required=True, help='Path to preprocessed CSV file')
    parser.add_argument('--output', type=str, default='results/clustering_outliers.csv', 
                       help='Path to output CSV file')
    parser.add_argument('--model', type=str, default='models/clustering_model.pkl', 
                       help='Path to save/load model')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for training')
    parser.add_argument('--eps', type=float, default=None, help='DBSCAN epsilon parameter')
    parser.add_argument('--min-samples', type=int, default=5, help='DBSCAN min_samples parameter')
    parser.add_argument('--kmeans', action='store_true', help='Use KMeans instead of DBSCAN')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters for KMeans')
    
    args = parser.parse_args()
    
    # Create model instance
    model = ClusteringModel(
        model_path=args.model,
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Training, prediction, or evaluation
    if args.train:
        print("=== Training Mode ===")
        model.train(df, sample_size=args.sample, use_kmeans=args.kmeans, n_clusters=args.clusters)
        model.save_model()
        
    if args.predict:
        print("=== Prediction Mode ===")
        if not model.model and not args.train:
            model.load_model()
        result_df = model.predict(df)
        model.save_predictions(result_df, args.output)
        
    if args.evaluate:
        print("=== Evaluation Mode ===")
        if not model.model and not args.train:
            model.load_model()
        metrics = model.evaluate(df)