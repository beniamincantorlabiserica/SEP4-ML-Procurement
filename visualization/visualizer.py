#!/usr/bin/env python3
"""
TED Data Visualization Module
This module provides simple, straightforward visualization capabilities for EU procurement data.
It creates clear charts for understanding outlier detection results.

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TEDVisualizer:
    """Class for creating visualizations of TED procurement data analysis"""
    
    def __init__(self, output_dir="visualizations", dpi=100, figsize=(10, 6)):
        """
        Initialize the visualizer
        
        Args:
            output_dir (str): Directory for saving visualizations
            dpi (int): Resolution for saved images
            figsize (tuple): Default figure size (width, height)
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization style
        sns.set_style("whitegrid")
        
        # Define colors for consistent look
        self.colors = {
            "normal": "#1f77b4",  # Blue
            "outlier": "#d62728"   # Red
        }
        
        # Set basic pyplot parameters
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
    
    def _format_currency(self, x, pos):
        """Format values as Euro currency with appropriate scale"""
        if x >= 1e9:
            return '€{:.1f}B'.format(x / 1e9)
        elif x >= 1e6:
            return '€{:.1f}M'.format(x / 1e6)
        elif x >= 1e3:
            return '€{:.1f}K'.format(x / 1e3)
        else:
            return '€{:.0f}'.format(x)
    
    def create_visualizations(self, df, base_filename="ted_analysis"):
        """
        Create a set of visualizations for procurement data analysis
        
        Args:
            df (pd.DataFrame): DataFrame with outlier predictions
            base_filename (str): Base name for output files
            
        Returns:
            list: Paths to saved visualization files
        """
        if df.empty:
            logger.warning("Empty dataframe provided, cannot create visualizations")
            return []
        
        saved_files = []
        
        try:
            # 1. Value Distribution
            value_path = self.plot_value_distribution(df, f"{base_filename}_values.png")
            if value_path:
                saved_files.append(value_path)
            
            # 2. Outlier Analysis
            if 'is_outlier' in df.columns:
                outlier_path = self.plot_outlier_analysis(df, f"{base_filename}_outliers.png")
                if outlier_path:
                    saved_files.append(outlier_path)
            
            # 3. Country Distribution
            country_field = None
            for col in ['organisation-country-buyer', 'country']:
                if col in df.columns:
                    country_field = col
                    break
                    
            if country_field:
                country_path = self.plot_country_distribution(df, country_field, f"{base_filename}_countries.png")
                if country_path:
                    saved_files.append(country_path)
            
            # 4. Summary Report
            report_path = self.create_summary_report(df, f"{base_filename}_report.png")
            if report_path:
                saved_files.append(report_path)
            
            logger.info(f"Created {len(saved_files)} visualizations")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return saved_files
    
    def plot_value_distribution(self, df, filename):
        """
        Create visualization of procurement value distribution
        
        Args:
            df (pd.DataFrame): DataFrame with procurement data
            filename (str): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Find value column
            value_col = None
            for col in ['total-value-eur', 'total-value-eur-capped', 'value_eur', 'total-value']:
                if col in df.columns:
                    value_col = col
                    break
            
            if not value_col:
                logger.warning("No value column found for value distribution plot")
                return None
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # Filter to valid values only
            plot_df = df[df[value_col] > 0].copy()
            
            # Histogram with log scale
            if 'is_outlier' in df.columns:
                # Separate histograms for normal and outlier values
                sns.histplot(
                    data=plot_df[~plot_df['is_outlier']], 
                    x=value_col,
                    bins=20,
                    log_scale=True, 
                    color=self.colors['normal'],
                    alpha=0.7,
                    label='Normal',
                    ax=ax1
                )
                
                sns.histplot(
                    data=plot_df[plot_df['is_outlier']], 
                    x=value_col,
                    bins=20,
                    log_scale=True, 
                    color=self.colors['outlier'],
                    alpha=0.7,
                    label='Outlier',
                    ax=ax1
                )
                
                ax1.legend()
            else:
                # Single histogram
                sns.histplot(
                    data=plot_df, 
                    x=value_col,
                    bins=30,
                    log_scale=True, 
                    color=self.colors['normal'],
                    ax=ax1
                )
            
            ax1.set_title('Value Distribution (Log Scale)')
            ax1.set_xlabel('Value (EUR)')
            ax1.set_ylabel('Count')
            
            # Boxplot of values
            if 'is_outlier' in df.columns:
                # Create boxplot by outlier status
                plot_df['Status'] = plot_df['is_outlier'].map({True: 'Outlier', False: 'Normal'})
                sns.boxplot(
                    x='Status',
                    y=value_col,
                    data=plot_df,
                    palette={
                        'Normal': self.colors['normal'],
                        'Outlier': self.colors['outlier']
                    },
                    ax=ax2
                )
            else:
                # Single boxplot
                sns.boxplot(
                    y=value_col,
                    data=plot_df,
                    color=self.colors['normal'],
                    ax=ax2
                )
            
            ax2.set_title('Value Range')
            ax2.set_ylabel('Value (EUR)')
            ax2.set_yscale('log')
            
            # Format y-axis as currency
            for axis in [ax1.xaxis, ax2.yaxis]:
                axis.set_major_formatter(plt.FuncFormatter(self._format_currency))
            
            plt.suptitle('Procurement Value Analysis', fontsize=16)
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved value distribution plot to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating value distribution plot: {e}")
            return None
    
    def plot_outlier_analysis(self, df, filename):
        """
        Create visualization focused on outlier analysis
        
        Args:
            df (pd.DataFrame): DataFrame with outlier predictions
            filename (str): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if 'is_outlier' not in df.columns:
                logger.warning("No is_outlier column found for outlier analysis plot")
                return None
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # 1. Pie chart of outlier distribution
            outlier_count = df['is_outlier'].sum()
            normal_count = len(df) - outlier_count
            
            ax1.pie(
                [normal_count, outlier_count],
                labels=['Normal', 'Outlier'],
                autopct='%1.1f%%',
                colors=[self.colors['normal'], self.colors['outlier']],
                startangle=90,
                explode=(0, 0.1)
            )
            
            ax1.set_title('Outlier Distribution')
            
            # 2. Anomaly score plot if available
            if 'anomaly_score' in df.columns:
                # Find value column for scatter plot
                value_col = None
                for col in ['total-value-eur', 'total-value-eur-capped', 'value_eur', 'total-value']:
                    if col in df.columns:
                        value_col = col
                        break
                
                if value_col:
                    # Scatter plot of value vs anomaly score
                    ax2.scatter(
                        df[~df['is_outlier']]['anomaly_score'],
                        df[~df['is_outlier']][value_col],
                        color=self.colors['normal'],
                        alpha=0.6,
                        label='Normal',
                        s=20
                    )
                    
                    ax2.scatter(
                        df[df['is_outlier']]['anomaly_score'],
                        df[df['is_outlier']][value_col],
                        color=self.colors['outlier'],
                        alpha=0.8,
                        label='Outlier',
                        s=40,
                        marker='x'
                    )
                    
                    ax2.set_title('Value vs Anomaly Score')
                    ax2.set_xlabel('Anomaly Score')
                    ax2.set_ylabel('Value (EUR)')
                    ax2.set_yscale('log')
                    ax2.legend()
                    
                    # Format y-axis as currency
                    ax2.yaxis.set_major_formatter(plt.FuncFormatter(self._format_currency))
                    
                    # Add threshold line if possible
                    if df['is_outlier'].sum() > 0:
                        threshold = df[df['is_outlier']]['anomaly_score'].min()
                        ax2.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
                        ax2.legend()
                else:
                    # Simple histogram of anomaly scores
                    sns.histplot(
                        data=df,
                        x='anomaly_score',
                        hue='is_outlier',
                        palette=[self.colors['normal'], self.colors['outlier']],
                        bins=20,
                        ax=ax2
                    )
                    
                    ax2.set_title('Anomaly Score Distribution')
                    ax2.set_xlabel('Anomaly Score')
                    ax2.set_ylabel('Count')
            else:
                # Show stats if no anomaly score
                ax2.axis('off')
                
                stats_text = [
                    f"Total Records: {len(df):,}",
                    f"Normal Records: {normal_count:,} ({normal_count/len(df)*100:.1f}%)",
                    f"Outlier Records: {outlier_count:,} ({outlier_count/len(df)*100:.1f}%)"
                ]
                
                # Add value stats if available
                value_col = None
                for col in ['total-value-eur', 'total-value-eur-capped', 'value_eur', 'total-value']:
                    if col in df.columns:
                        value_col = col
                        break
                
                if value_col:
                    normal_avg = df[~df['is_outlier']][value_col].mean()
                    outlier_avg = df[df['is_outlier']][value_col].mean() if outlier_count > 0 else 0
                    
                    stats_text.extend([
                        f"",
                        f"Avg Normal Value: {self._format_currency(normal_avg, 0)}",
                        f"Avg Outlier Value: {self._format_currency(outlier_avg, 0)}",
                        f"Outlier/Normal Ratio: {outlier_avg/normal_avg:.1f}x" if normal_avg > 0 else ""
                    ])
                
                ax2.text(
                    0.5, 0.5,
                    '\n'.join(stats_text),
                    ha='center',
                    va='center',
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='#f9f9f9', alpha=0.5)
                )
            
            plt.suptitle('Procurement Outlier Analysis', fontsize=16)
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved outlier analysis plot to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating outlier analysis plot: {e}")
            return None
    
    def plot_country_distribution(self, df, country_field, filename):
        """
        Create visualization of procurement data by country
        
        Args:
            df (pd.DataFrame): DataFrame with procurement data
            country_field (str): Name of column containing country data
            filename (str): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if country_field not in df.columns:
                logger.warning(f"Column {country_field} not found for country distribution plot")
                return None
            
            # Get top countries by count
            top_countries = df[country_field].value_counts().head(10)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # 1. Count by country
            sns.barplot(
                x=top_countries.index,
                y=top_countries.values,
                palette='viridis',
                ax=ax1
            )
            
            ax1.set_title('Record Count by Country')
            ax1.set_xlabel('')
            ax1.set_ylabel('Number of Records')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # 2. Outlier percentage by country
            if 'is_outlier' in df.columns:
                # Calculate percentage of outliers by country
                country_data = df.groupby(country_field)['is_outlier'].agg(['count', 'sum'])
                country_data['percentage'] = country_data['sum'] / country_data['count'] * 100
                
                # Sort by count and get top countries
                country_data = country_data.sort_values('count', ascending=False).head(10)
                
                sns.barplot(
                    x=country_data.index,
                    y=country_data['percentage'],
                    palette='rocket_r',
                    ax=ax2
                )
                
                # Add horizontal line for overall percentage
                overall_pct = df['is_outlier'].mean() * 100
                ax2.axhline(
                    y=overall_pct,
                    color='red',
                    linestyle='--',
                    label=f'Overall: {overall_pct:.1f}%'
                )
                
                ax2.set_title('Outlier Percentage by Country')
                ax2.set_xlabel('')
                ax2.set_ylabel('Outlier Percentage (%)')
                ax2.legend()
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                # Value distribution by country
                value_col = None
                for col in ['total-value-eur', 'total-value-eur-capped', 'value_eur', 'total-value']:
                    if col in df.columns:
                        value_col = col
                        break
                
                if value_col:
                    # Calculate average value by country
                    country_values = df.groupby(country_field)[value_col].mean().sort_values(ascending=False).head(10)
                    
                    sns.barplot(
                        x=country_values.index,
                        y=country_values.values,
                        palette='viridis',
                        ax=ax2
                    )
                    
                    ax2.set_title('Average Value by Country')
                    ax2.set_xlabel('')
                    ax2.set_ylabel('Average Value (EUR)')
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                    
                    # Format y-axis as currency
                    ax2.yaxis.set_major_formatter(plt.FuncFormatter(self._format_currency))
                else:
                    ax2.set_title('No value data available')
                    ax2.axis('off')
            
            plt.suptitle('Procurement Analysis by Country', fontsize=16)
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved country distribution plot to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating country distribution plot: {e}")
            return None
    
    def create_summary_report(self, df, filename):
        """
        Create a summary report visualization
        
        Args:
            df (pd.DataFrame): DataFrame with procurement data
            filename (str): Output filename
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Create figure with 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Find value column
            value_col = None
            for col in ['total-value-eur', 'total-value-eur-capped', 'value_eur', 'total-value']:
                if col in df.columns:
                    value_col = col
                    break
            
            # 1. Value histogram (top-left)
            ax1 = axes[0, 0]
            
            if value_col:
                # Filter to valid values
                plot_df = df[df[value_col] > 0].copy()
                
                if 'is_outlier' in df.columns:
                    # Separate histograms for normal and outlier values
                    sns.histplot(
                        data=plot_df[~plot_df['is_outlier']], 
                        x=value_col,
                        bins=20,
                        log_scale=True, 
                        color=self.colors['normal'],
                        alpha=0.7,
                        label='Normal',
                        ax=ax1
                    )
                    
                    sns.histplot(
                        data=plot_df[plot_df['is_outlier']], 
                        x=value_col,
                        bins=20,
                        log_scale=True, 
                        color=self.colors['outlier'],
                        alpha=0.7,
                        label='Outlier',
                        ax=ax1
                    )
                    
                    ax1.legend()
                else:
                    # Single histogram
                    sns.histplot(
                        data=plot_df, 
                        x=value_col,
                        bins=30,
                        log_scale=True, 
                        color=self.colors['normal'],
                        ax=ax1
                    )
                
                ax1.set_title('Value Distribution')
                ax1.set_xlabel('Value (EUR, log scale)')
                ax1.set_ylabel('Count')
                
                # Format x-axis as currency
                ax1.xaxis.set_major_formatter(plt.FuncFormatter(self._format_currency))
            else:
                ax1.set_title('No value data available')
                ax1.axis('off')
            
            # 2. Country bar chart (top-right)
            ax2 = axes[0, 1]
            
            # Find country field
            country_field = None
            for col in ['organisation-country-buyer', 'country']:
                if col in df.columns:
                    country_field = col
                    break
            
            if country_field:
                # Get top countries by count
                top_countries = df[country_field].value_counts().head(10)
                
                sns.barplot(
                    x=top_countries.index,
                    y=top_countries.values,
                    palette='viridis',
                    ax=ax2
                )
                
                ax2.set_title('Top Countries')
                ax2.set_xlabel('')
                ax2.set_ylabel('Number of Records')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.set_title('No country data available')
                ax2.axis('off')
            
            # 3. Outlier analysis (bottom-left)
            ax3 = axes[1, 0]
            
            if 'is_outlier' in df.columns:
                # Pie chart of outlier distribution
                outlier_count = df['is_outlier'].sum()
                normal_count = len(df) - outlier_count
                
                ax3.pie(
                    [normal_count, outlier_count],
                    labels=['Normal', 'Outlier'],
                    autopct='%1.1f%%',
                    colors=[self.colors['normal'], self.colors['outlier']],
                    startangle=90,
                    explode=(0, 0.1)
                )
                
                ax3.set_title('Outlier Distribution')
            else:
                ax3.set_title('No outlier data available')
                ax3.axis('off')
            
            # 4. Summary statistics text (bottom-right)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Prepare statistics text
            stats = [
                f"SUMMARY STATISTICS",
                f"",
                f"Total Records: {len(df):,}"
            ]
            
            if 'is_outlier' in df.columns:
                outlier_count = df['is_outlier'].sum()
                outlier_pct = outlier_count / len(df) * 100
                stats.extend([
                    f"Outliers: {outlier_count:,} ({outlier_pct:.1f}%)",
                    f"Normal Records: {len(df) - outlier_count:,} ({100 - outlier_pct:.1f}%)"
                ])
            
            if value_col:
                valid_values = df[df[value_col] > 0][value_col]
                stats.extend([
                    f"",
                    f"Value Statistics:",
                    f"Minimum: {self._format_currency(valid_values.min(), 0)}",
                    f"Maximum: {self._format_currency(valid_values.max(), 0)}",
                    f"Average: {self._format_currency(valid_values.mean(), 0)}",
                    f"Median: {self._format_currency(valid_values.median(), 0)}"
                ])
                
                if 'is_outlier' in df.columns and outlier_count > 0:
                    normal_avg = df[~df['is_outlier']][value_col].mean()
                    outlier_avg = df[df['is_outlier']][value_col].mean()
                    stats.extend([
                        f"",
                        f"Average Normal Value: {self._format_currency(normal_avg, 0)}",
                        f"Average Outlier Value: {self._format_currency(outlier_avg, 0)}",
                        f"Outlier/Normal Ratio: {outlier_avg/normal_avg:.1f}x"
                    ])
            
            # Add timestamp
            stats.extend([
                f"",
                f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ])
            
            # Display statistics
            ax4.text(
                0.5, 0.5,
                '\n'.join(stats),
                ha='center',
                va='center',
                transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f9f9f9', alpha=0.5)
            )
            
            plt.suptitle('EU Procurement Analysis Summary', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
            
            # Save figure
            output_path = os.path.join(self.output_dir, filename)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved summary report to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            return None

# If run directly, perform a simple test
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='TED Data Visualization')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='Path to input CSV file with procurement data')
    parser.add_argument('--output', '-o', type=str, default='visualizations', 
                       help='Output directory for visualizations')
    parser.add_argument('--name', '-n', type=str, default='ted_analysis', 
                       help='Base name for output files')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        exit(1)
    
    # Create visualizer
    visualizer = TEDVisualizer(output_dir=args.output)
    
    # Load data
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} records from {args.input}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Create visualizations
    viz_files = visualizer.create_visualizations(df, args.name)
    
    # Print results
    if viz_files:
        print(f"\nCreated {len(viz_files)} visualizations:")
        for file in viz_files:
            print(f" - {file}")
    else:
        print("\nNo visualizations were created")