"""
DigiArvi Descriptive Analysis Script

This script performs comprehensive descriptive analysis on the DigiArvi dataset,
including statistical summaries and visualizations of key variables.

The analysis includes:
- Basic descriptive statistics for numerical variables
- Distribution analysis for categorical variables
- Correlation analysis between variables
- Visualizations of score distributions and relationships
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import logging
from typing import Dict, Tuple, Any

# Configuration
RESULTS_DIR = 'results_descriptive'
DATA_FILE = 'CPFQ_REVERSED_USETHIS.xls'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Plot settings
PLOT_STYLE = 'whitegrid'
PLOT_CONTEXT = 'paper'

# Color palettes for different visualizations
DIST_PALETTE = 'viridis'  # For distributions
CAT_PALETTE = 'Set3'      # For categorical plots
CORR_PALETTE = 'coolwarm'  # For correlation heatmap
COMP_PALETTE = 'husl'      # For comparison plots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create results directory if it doesn't exist
try:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info(f"Created results directory: {RESULTS_DIR}")
except Exception as e:
    logging.error(f"Failed to create results directory: {e}")
    raise

def load_data() -> pd.DataFrame:
    """Load the DigiArvi dataset from Excel file.
    
    Returns:
        pd.DataFrame: The loaded dataset
    
    Raises:
        FileNotFoundError: If the data file is not found
        Exception: For other errors during data loading
    """
    try:
        df = pd.read_excel(DATA_FILE, engine='xlrd')
        logging.info(f"Successfully loaded data from {DATA_FILE}")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found: {DATA_FILE}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def generate_descriptive_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate comprehensive descriptive statistics"""
    # Numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Enhanced numerical statistics
    basic_stats = df[numeric_cols].describe().round(2)
    additional_stats = pd.DataFrame({
        'skewness': df[numeric_cols].skew(),
        'kurtosis': df[numeric_cols].kurtosis(),
        'median': df[numeric_cols].median(),
        'iqr': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
        'missing': df[numeric_cols].isnull().sum(),
        'missing_pct': (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
    })
    numeric_stats = pd.concat([basic_stats, additional_stats.T])
    
    # Categorical distributions with percentages
    categorical_cols = ['sex', 'school_lang', 'home_lang', 'strong_lang', 'friend_lang', 'grade']
    cat_distributions = {}
    for col in categorical_cols:
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True).round(3) * 100
        cat_distributions[col] = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages
        })
    
    # Calculate correlations
    correlations = df[numeric_cols].corr().round(3)
    
    return numeric_stats, cat_distributions, correlations

def create_score_distributions(df: pd.DataFrame) -> None:
    """Create distribution plots for SCORE_SRF and SCORE_F3.2.
    
    Args:
        df: Input DataFrame containing score columns
    """
    plt.figure(figsize=(15, 6))
    score_cols = ['SCORE_SRF', 'SCORE_F3.2']
    colors = sns.color_palette(DIST_PALETTE, n_colors=2)
    for i, (col, color) in enumerate(zip(score_cols, colors)):
        plt.subplot(1, 2, i+1)
        sns.histplot(data=df, x=col, kde=True, color=color)
        plt.title(f'{col} Distribution', fontsize=12, pad=15)
        plt.axvline(df[col].mean(), color='crimson', linestyle='--', label='Mean', linewidth=2)
        plt.axvline(df[col].median(), color='forestgreen', linestyle='--', label='Median', linewidth=2)
        plt.legend(fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'score_distributions.png')
    plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()
    logging.info(f"Saved score distributions plot to {save_path}")

def create_categorical_plots(df: pd.DataFrame) -> None:
    """Create distribution plots for categorical variables.
    
    Args:
        df: Input DataFrame containing categorical columns
    """
    categorical_cols = ['sex', 'school_lang', 'home_lang', 'strong_lang', 'friend_lang', 'grade']
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    colors = sns.color_palette(CAT_PALETTE, n_colors=len(categorical_cols))
    for i, (col, ax, color) in enumerate(zip(categorical_cols, axes.flat, colors)):
        sns.countplot(data=df, x=col, ax=ax, color=color)
        ax.set_title(f'{col.replace("_", " ").title()} Distribution')
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = len(df[col].dropna())
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'categorical_distributions.png')
    plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.close()
    logging.info(f"Saved categorical distributions plot to {save_path}")

def create_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create correlation heatmap for selected variables.
    
    Args:
        df: Input DataFrame containing variables for correlation analysis
    """
    # Select variables for correlation analysis
    selected_vars = [
        'SCORE_SRF', 'SCORE_F3.2',  # Score variables
        'T10_TOTAL_theta_25',  # T10 total score
        'cpfq_1', 'cpfq_2', 'cpfq_3', 'cpfq_4', 'cpfq_7', 'cpfq_8', 'cpfq_10', 'cpfq_11', 'cpfq_14',
        'cpfq_5_rev', 'cpfq_6_rev', 'cpfq_9_rev', 'cpfq_12_rev', 'cpfq_13_rev', 'cpfq_15_rev', 'cpfq_16_rev', 'cpfq_17_rev', 'cpfq_18_rev'
    ]
    
    # Create a copy of the dataframe with selected variables
    df_selected = df[selected_vars].copy()
    
    # Calculate correlation matrix
    corr_matrix = df_selected.corr()
    
    # Create figure with appropriate size
    plt.figure(figsize=(20, 16))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Create heatmap with improved styling
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap=CORR_PALETTE, 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'},
                annot_kws={'size': 8})
    
    plt.title('Correlation Matrix of Selected Variables', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', size=8)
    plt.yticks(rotation=0, size=8)
    
    save_path = os.path.join(RESULTS_DIR, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved correlation heatmap to {save_path}")

def create_score_comparisons(df: pd.DataFrame) -> None:
    """Create box plots comparing scores across different categorical variables.
    
    Args:
        df: Input DataFrame containing score and categorical columns
    """
    for cat_var in ['sex', 'grade', 'school_lang']:
        plt.figure(figsize=(12, 6))
        colors = sns.color_palette(COMP_PALETTE, n_colors=len(df[cat_var].unique()))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x=cat_var, y='SCORE_SRF', palette=colors)
        plt.title(f'SCORE_SRF by {cat_var.replace("_", " ").title()}', fontsize=11)
        plt.xticks(rotation=30, ha='right')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x=cat_var, y='SCORE_F3.2', palette=colors)
        plt.title(f'SCORE_F3.2 by {cat_var.replace("_", " ").title()}', fontsize=11)
        plt.xticks(rotation=30, ha='right')
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'scores_by_{cat_var}.png')
        plt.savefig(save_path, dpi=FIGURE_DPI)
        plt.close()
        logging.info(f"Saved score comparison plot to {save_path}")

def create_visualizations(df: pd.DataFrame) -> None:
    """Create all visualizations for the analysis.
    
    Args:
        df: Input DataFrame containing all required columns
    """
    # Set global plotting style
    sns.set_style(PLOT_STYLE)
    sns.set_context(PLOT_CONTEXT)
    
    try:
        create_score_distributions(df)
        create_categorical_plots(df)
        create_correlation_heatmap(df)
        create_score_comparisons(df)
        logging.info("Successfully created all visualizations")
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
        raise

def save_results(numeric_stats: pd.DataFrame, 
               cat_distributions: Dict[str, pd.DataFrame], 
               correlations: pd.DataFrame) -> None:
    """Save analysis results to a text file.
    
    Args:
        numeric_stats: DataFrame containing numerical statistics
        cat_distributions: Dictionary of categorical variable distributions
        correlations: DataFrame containing correlation matrix
    """
    output_file = os.path.join(RESULTS_DIR, 'descriptive_statistics.txt')
    try:
        with open(output_file, 'w') as f:
            # Write numerical statistics
            f.write("COMPREHENSIVE NUMERICAL STATISTICS\n")
            f.write("================================\n\n")
            f.write(numeric_stats.to_string())
            f.write("\n\n")
            
            # Write categorical distributions
            f.write("CATEGORICAL VARIABLE DISTRIBUTIONS\n")
            f.write("================================\n\n")
            for var_name, dist in cat_distributions.items():
                f.write(f"{var_name.replace('_', ' ').title()} Distribution:\n")
                f.write(dist.to_string())
                f.write("\n\n")
            
            # Write correlation matrix
            f.write("CORRELATION MATRIX\n")
            f.write("=================\n\n")
            f.write(correlations.to_string())
            f.write("\n\n")
            
            # Write correlation interpretation guide
            f.write("CORRELATION INTERPRETATION GUIDE\n")
            f.write("============================\n\n")
            f.write("- Values close to 1 indicate strong positive correlation")
            f.write("\n- Values close to -1 indicate strong negative correlation")
            f.write("\n- Values close to 0 indicate weak or no correlation")
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        raise

def main() -> None:
    """Main function to run the descriptive analysis."""
    try:
        # Load data
        logging.info("Loading data...")
        df = load_data()
        
        # Generate statistics
        logging.info("Generating descriptive statistics...")
        numeric_stats, cat_distributions, correlations = generate_descriptive_stats(df)
        
        # Create visualizations
        logging.info("Creating visualizations...")
        create_visualizations(df)
        
        # Save results
        logging.info("Saving results...")
        save_results(numeric_stats, cat_distributions, correlations)
        
        # Print summary
        print("\nAnalysis complete!")
        print(f"Results have been saved to the '{RESULTS_DIR}' directory:")
        for i, file in enumerate([
            'descriptive_statistics.txt',
            'score_distributions.png',
            'categorical_distributions.png',
            'correlation_heatmap.png',
            'scores_by_sex.png',
            'scores_by_grade.png',
            'scores_by_school_lang.png'
        ], 1):
            print(f"{i}. {os.path.join(RESULTS_DIR, file)}")
    
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
