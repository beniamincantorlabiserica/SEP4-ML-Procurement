#!/usr/bin/env python3
"""
Example implementation of a ClusteringPipeline for the EU Procurement Monitoring System.
This extension adds clustering-based outlier detection to the system.

"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from ted_ml_pipeline import TEDMLPipeline
from visualization.visualizer import TEDVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClusteringPipeline(TEDMLPipeline):
    """
    Extension of TEDMLPipeline that uses clustering-based methods for outlier detection.
    Supports both DBSCAN and KMeans clustering algorithms.
    """
    
    def __init__(self, base_dir="."):
        """Initialize the clustering pipeline"""
        super().__init__(base_dir)
        
        # Add clustering-specific directories
        self.directories["clusters"] = os.path.join(base_dir, "clusters")
        os.makedirs(self.directories["clusters"], exist_ok=True)
        
        # Initialize clustering parameters with defaults
        self.clustering_params = {
            "method": "dbscan",  # 'dbscan' or 'kmeans'
            "eps": 0.5,          # For DBSCAN: maximum distance between samples
            "min_samples": 5,    # For DBSCAN: number of samples in neighborhood
            "n_clusters": 5      # For KMeans: number of clusters
        }
    
    def train_clustering_model(self, input_file=None, method="dbscan", **kwargs):
        """
        Train a clustering model for outlier detection
        
        Args:
            input_file (str): Optional path to training data file
            method (str): Clustering method ('dbscan' or 'kmeans')
            **kwargs: Additional parameters for the clustering algorithm
                - eps: DBSCAN epsilon parameter (default: 0.5)
                - min_samples: DBSCAN min_samples parameter (default: 5)
                - n_clusters: KMeans number of clusters (default: 5)
        
        Returns:
            dict: Dictionary with model information
        """
        logger.info(f"Training clustering model: method={method}")
        
        # Update clustering parameters
        self.clustering_params["method"] = method
        if "eps" in kwargs:
            self.clustering_params["eps"] = kwargs["eps"]
        if "min_samples" in kwargs:
            self.clustering_params["min_samples"] = kwargs["min_samples"]
        if "n_clusters" in kwargs:
            self.clustering_params["n_clusters"] = kwargs["n_clusters"]
        
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
                df = pd.read_csv(training_file)
                print(f"Loaded {len(df)} records with {len(df.columns)} features")
            except Exception as e:
                print(f"Error loading training data: {e}")
                return None
            
            # Step 3: Select and prepare features for clustering
            print("Preparing features for clustering...")
            
            # Get numerical features only
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Remove ID columns and other irrelevant features
            exclude_patterns = ['id', 'identifier', 'code', 'status', 'is_outlier']
            feature_cols = [col for col in num_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]
            
            # Ensure we have valid numerical data
            X = df[feature_cols].fillna(0).values
            
            # Scale the data
            print("Scaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Step 4: Train the clustering model
            print(f"Training {method.upper()} clustering model...")
            if method.lower() == "dbscan":
                # Train DBSCAN model
                eps = self.clustering_params["eps"]
                min_samples = self.clustering_params["min_samples"]
                
                print(f"Parameters: eps={eps}, min_samples={min_samples}")
                model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                
            elif method.lower() == "kmeans":
                # Train KMeans model
                n_clusters = self.clustering_params["n_clusters"]
                
                print(f"Parameters: n_clusters={n_clusters}")
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
            else:
                print(f"Error: Unknown clustering method '{method}'")
                return None
            
            # Fit the model
            model.fit(X_scaled)
            
            # Step 5: Process clustering results
            labels = model.labels_
            
            # For DBSCAN: -1 indicates outliers
            if method.lower() == "dbscan":
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                outlier_pct = n_outliers / len(labels) * 100
                
                print(f"DBSCAN results: {n_clusters} clusters, {n_outliers} outliers ({outlier_pct:.2f}%)")
                
                # Add cluster labels to the dataframe
                df["cluster"] = labels
                df["is_outlier"] = labels == -1
                
            # For KMeans: find outliers as points far from their centroids
            else:
                # Calculate distance to assigned centroid
                distances = np.min(
                    np.sqrt(np.sum((X_scaled - model.cluster_centers_[labels.reshape(-1, 1)])**2, axis=2)),
                    axis=1
                )
                
                # Define outliers as points with distance > threshold (95th percentile)
                threshold = np.percentile(distances, 95)
                outliers = distances > threshold
                n_outliers = sum(outliers)
                outlier_pct = n_outliers / len(distances) * 100
                
                print(f"KMeans results: {n_clusters} clusters, {n_outliers} outliers ({outlier_pct:.2f}%)")
                
                # Add cluster labels and outlier flags to the dataframe
                df["cluster"] = labels
                df["is_outlier"] = outliers
                df["distance"] = distances
            
            # Also add numerical anomaly score (0-1 scale)
            if method.lower() == "dbscan":
                # For DBSCAN: Points marked as outliers get max score, others based on nearest neighbors
                from sklearn.neighbors import NearestNeighbors
                
                # Calculate distance to k-nearest neighbors for non-outliers
                nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
                distances, _ = nbrs.kneighbors(X_scaled)
                
                # Average distance to k nearest neighbors as score
                avg_distances = np.mean(distances, axis=1)
                max_dist = np.max(avg_distances)
                
                # Normalize scores to 0-1 range
                scores = avg_distances / max_dist
                
                # Ensure outliers have highest scores
                scores[labels == -1] = 1.0
                
            else:
                # For KMeans: Normalize distances to 0-1 range
                scores = distances / np.max(distances)
            
            # Add anomaly scores to dataframe
            df["anomaly_score"] = scores
            
            # Step 6: Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.directories["clusters"], f"cluster_results_{timestamp}.csv")
            
            print(f"Saving clustering results to {results_file}")
            df.to_csv(results_file, index=False)
            
            # Save model information
            model_info = {
                "method": method,
                "parameters": self.clustering_params,
                "feature_columns": feature_cols,
                "scaler": scaler,
                "model": model,
                "results_file": results_file,
                "timestamp": timestamp
            }
            
            print("Clustering model training completed successfully")
            
            # Generate visualizations if requested
            if kwargs.get("visualize", False):
                print("Generating visualizations...")
                
                visualizer = TEDVisualizer(output_dir=self.directories["visualizations"])
                viz_files = visualizer.create_visualizations(
                    df, 
                    base_filename=f"clustering_{method}_{timestamp}"
                )
                
                if viz_files:
                    print(f"Generated {len(viz_files)} visualizations")
                    model_info["visualizations"] = viz_files
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error during clustering model training: {e}")
            print(f"Error: {str(e)}")
            return None
    
    def predict_with_clustering(self, input_file=None, method="dbscan", visualize=False, **kwargs):
        """
        Detect outliers using a clustering approach
        
        Args:
            input_file (str): Path to input data file
            method (str): Clustering method ('dbscan' or 'kmeans')
            visualize (bool): Whether to generate visualizations
            **kwargs: Additional parameters for the clustering algorithm
            
        Returns:
            dict: Dictionary with prediction results
        """
        logger.info(f"Detecting outliers with clustering: method={method}")
        
        # Update parameters for this run only (don't change stored parameters)
        clustering_params = dict(self.clustering_params)
        if "eps" in kwargs:
            clustering_params["eps"] = kwargs["eps"]
        if "min_samples" in kwargs:
            clustering_params["min_samples"] = kwargs["min_samples"]
        if "n_clusters" in kwargs:
            clustering_params["n_clusters"] = kwargs["n_clusters"]
        
        try:
            # Step 1: Load the dataset
            if not input_file or not os.path.exists(input_file):
                print(f"Error: Input file {input_file} does not exist")
                return None
                
            print(f"Loading dataset: {input_file}")
            try:
                df = pd.read_csv(input_file)
                print(f"Loaded {len(df)} records with {len(df.columns)} features")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                return None
            
            # Step 2: Select and prepare features
            print("Preparing features for clustering...")
            
            # Get numerical features only
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Remove ID columns and other irrelevant features
            exclude_patterns = ['id', 'identifier', 'code', 'status', 'is_outlier']
            feature_cols = [col for col in num_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]
            
            # Ensure we have valid numerical data
            X = df[feature_cols].fillna(0).values
            
            # Scale the data
            print("Scaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Step 3: Apply clustering algorithm
            print(f"Applying {method.upper()} clustering...")
            if method.lower() == "dbscan":
                # Apply DBSCAN
                eps = clustering_params["eps"]
                min_samples = clustering_params["min_samples"]
                
                print(f"Parameters: eps={eps}, min_samples={min_samples}")
                model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                
            elif method.lower() == "kmeans":
                # Apply KMeans
                n_clusters = clustering_params["n_clusters"]
                
                print(f"Parameters: n_clusters={n_clusters}")
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
            else:
                print(f"Error: Unknown clustering method '{method}'")
                return None
            
            # Fit the model
            model.fit(X_scaled)
            
            # Step 4: Process clustering results
            labels = model.labels_
            
            # For DBSCAN: -1 indicates outliers
            if method.lower() == "dbscan":
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1)
                outlier_pct = n_outliers / len(labels) * 100
                
                print(f"DBSCAN results: {n_clusters} clusters, {n_outliers} outliers ({outlier_pct:.2f}%)")
                
                # Add cluster labels to the dataframe
                df["cluster"] = labels
                df["is_outlier"] = labels == -1
                
            # For KMeans: find outliers as points far from their centroids
            else:
                # Calculate distance to assigned centroid
                distances = np.min(
                    np.sqrt(np.sum((X_scaled - model.cluster_centers_[labels.reshape(-1, 1)])**2, axis=2)),
                    axis=1
                )
                
                # Define outliers as points with distance > threshold (95th percentile)
                threshold = np.percentile(distances, 95)
                outliers = distances > threshold
                n_outliers = sum(outliers)
                outlier_pct = n_outliers / len(distances) * 100
                
                print(f"KMeans results: {n_clusters} clusters, {n_outliers} outliers ({outlier_pct:.2f}%)")
                
                # Add cluster labels and outlier flags to the dataframe
                df["cluster"] = labels
                df["is_outlier"] = outliers
                df["distance"] = distances
            
            # Also add numerical anomaly score (0-1 scale)
            if method.lower() == "dbscan":
                # For DBSCAN: Points marked as outliers get max score, others based on nearest neighbors
                from sklearn.neighbors import NearestNeighbors
                
                # Calculate distance to k-nearest neighbors for non-outliers
                nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
                distances, _ = nbrs.kneighbors(X_scaled)
                
                # Average distance to k nearest neighbors as score
                avg_distances = np.mean(distances, axis=1)
                max_dist = np.max(avg_distances)
                
                # Normalize scores to 0-1 range
                scores = avg_distances / max_dist
                
                # Ensure outliers have highest scores
                scores[labels == -1] = 1.0
                
            else:
                # For KMeans: Normalize distances to 0-1 range
                scores = distances / np.max(distances)
            
            # Add anomaly scores to dataframe
            df["anomaly_score"] = scores
            
            # Step 5: Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.directories["output"], f"cluster_outliers_{timestamp}.csv")
            
            print(f"Saving results to {results_file}")
            df.to_csv(results_file, index=False)
            
            # Step 6: Generate visualizations if requested
            viz_files = []
            if visualize:
                print("Generating visualizations...")
                
                visualizer = TEDVisualizer(output_dir=self.directories["visualizations"])
                viz_files = visualizer.create_visualizations(
                    df, 
                    base_filename=f"clustering_{method}_{timestamp}"
                )
                
                if viz_files:
                    print(f"Generated {len(viz_files)} visualizations")
            
            # Prepare result information
            result = {
                "method": method,
                "parameters": clustering_params,
                "results_file": results_file,
                "record_count": len(df),
                "outlier_count": n_outliers,
                "outlier_percentage": outlier_pct,
                "visualizations": viz_files,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"Clustering-based outlier detection completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during clustering-based outlier detection: {e}")
            print(f"Error: {str(e)}")
            return None

# Example usage when run directly
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='EU Procurement Monitoring - Clustering Analysis')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a clustering model')
    train_parser.add_argument('--input', type=str, help='Path to input CSV file')
    train_parser.add_argument('--method', type=str, choices=['dbscan', 'kmeans'], 
                            default='dbscan', help='Clustering method')
    train_parser.add_argument('--eps', type=float, default=0.5, 
                            help='DBSCAN: distance threshold')
    train_parser.add_argument('--min-samples', type=int, default=5, 
                            help='DBSCAN: minimum samples in neighborhood')
    train_parser.add_argument('--n-clusters', type=int, default=5, 
                            help='KMeans: number of clusters')
    train_parser.add_argument('--visualize', action='store_true', 
                            help='Generate visualizations')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Detect outliers using clustering')
    predict_parser.add_argument('--input', type=str, required=True,
                              help='Path to input CSV file')
    predict_parser.add_argument('--method', type=str, choices=['dbscan', 'kmeans'], 
                              default='dbscan', help='Clustering method')
    predict_parser.add_argument('--eps', type=float, default=0.5, 
                              help='DBSCAN: distance threshold')
    predict_parser.add_argument('--min-samples', type=int, default=5, 
                              help='DBSCAN: minimum samples in neighborhood')
    predict_parser.add_argument('--n-clusters', type=int, default=5, 
                              help='KMeans: number of clusters')
    predict_parser.add_argument('--visualize', action='store_true', 
                              help='Generate visualizations')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = ClusteringPipeline()
    
    # Execute command
    if args.command == 'train':
        if args.method == 'dbscan':
            pipeline.train_clustering_model(
                input_file=args.input,
                method=args.method,
                eps=args.eps,
                min_samples=args.min_samples,
                visualize=args.visualize
            )
        else:  # kmeans
            pipeline.train_clustering_model(
                input_file=args.input,
                method=args.method,
                n_clusters=args.n_clusters,
                visualize=args.visualize
            )
    elif args.command == 'predict':
        if args.method == 'dbscan':
            pipeline.predict_with_clustering(
                input_file=args.input,
                method=args.method,
                eps=args.eps,
                min_samples=args.min_samples,
                visualize=args.visualize
            )
        else:  # kmeans
            pipeline.predict_with_clustering(
                input_file=args.input,
                method=args.method,
                n_clusters=args.n_clusters,
                visualize=args.visualize
            )
    else:
        parser.print_help()
        sys.exit(1)