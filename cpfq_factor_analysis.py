import os
import datetime
"""
CPFQ Factor Analysis Script

This script performs comprehensive factor analysis on the CPFQ dataset,
including data preparation, factor extraction, and visualization of results.

The analysis includes:
- Missing value analysis
- KMO test for sampling adequacy
- Bartlett's test of sphericity
- Factor analysis with varimax rotation
- Visualization of factor loadings and relationships
"""

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import logging
from typing import Dict, Tuple, Any, List

# Configuration
RESULTS_DIR = 'results_cpfq'
DATA_FILE = 'CPFQ_REVERSED_USETHIS.xls'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Plot settings
PLOT_STYLE = 'whitegrid'
PLOT_CONTEXT = 'paper'

# Color palettes for different visualizations
CORR_PALETTE = 'RdYlBu_r'     # Red-Yellow-Blue diverging palette for correlations
SCREE_PALETTE = 'viridis'     # Modern sequential palette for scree plot
FACTOR_PALETTE = 'tab20c'     # Categorical palette for factors
HEATMAP_PALETTE = 'coolwarm'  # Diverging palette for heatmaps

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

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare the CPFQ data by selecting specific items (1, 2, 3, 4, 7, 11, 12, 13, and 15).
    Takes into account reversed items (_rev suffix).
    
    Args:
        df: The loaded dataset
    
    Returns:
        Tuple containing:
            - pd.DataFrame: The prepared CPFQ data with selected items
            - List[str]: List of selected CPFQ column names
    """
    # Debug: print available columns
    print("Available columns in dataset:")
    cpfq_cols_available = [col for col in df.columns if col.startswith('cpfq_') and col != 'cpfq_Time']
    print("CPFQ columns:", sorted(cpfq_cols_available))
    
    selected_items = [1, 2, 3, 4, 7, 11, 12, 13, 15]
    cpfq_cols = []
    
    for item in selected_items:
        base_col = f'cpfq_{item}'
        rev_col = f'cpfq_{item}_rev'
        
        if base_col in df.columns:
            cpfq_cols.append(base_col)
        elif rev_col in df.columns:
            cpfq_cols.append(rev_col)
        else:
            print(f"Warning: Neither {base_col} nor {rev_col} found in dataset")
    
    if not cpfq_cols:
        raise ValueError("No requested CPFQ columns found in dataset")
    
    print("\nSelected columns for analysis:")
    print(cpfq_cols)
    
    cpfq_data = df[cpfq_cols].copy()
    return cpfq_data, cpfq_cols

def analyze_missing_and_pairs(cpfq_data: pd.DataFrame, cpfq_cols: List[str]) -> str:
    """Analyze missing values and pairs in the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    
    Returns:
        str: The analysis results as a string
    """
    print(f'\nAnalyzing CPFQ data with {len(cpfq_data)} total responses...\n')
    
    print('Detailed Analysis of Missing Values and Pairs:')
    print('=' * 50)
    
    # 1. Overall missing values
    print('\n1. Missing Values per Item:')
    missing_counts = cpfq_data.isnull().sum()
    total_responses = len(cpfq_data)
    missing_pct = (missing_counts / total_responses * 100).round(1)
    
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Total Possible': total_responses,
        'Missing %': missing_pct,
        'Available Responses': total_responses - missing_counts
    })
    print(missing_summary)
    
    # 2. Pairwise complete cases
    print('\n2. Pairwise Complete Cases Analysis:\n')
    print('Example Pair Counts (number of complete cases for each pair):\n')
    
    # Find most and least complete pairs
    n_pairs = []
    pair_info = []
    for i in range(len(cpfq_cols)):
        for j in range(i, len(cpfq_cols)):
            pair_complete = cpfq_data[[cpfq_cols[i], cpfq_cols[j]]].dropna().shape[0]
            n_pairs.append(pair_complete)
            pair_info.append(f'{cpfq_cols[i]} - {cpfq_cols[j]}: {pair_complete} responses ({(pair_complete/total_responses*100):.1f}%)')
            
            if i == j == 0:  # First variable with itself (most complete)
                print('Most Complete Pairs:')
                print(f'{cpfq_cols[i]} - {cpfq_cols[j]}: {pair_complete} responses ({(pair_complete/total_responses*100):.1f}%)')
            
            if pair_complete == min(n_pairs):  # Least complete pair
                print('\nLeast Complete Pairs:')
                print(f'{cpfq_cols[i]} - {cpfq_cols[j]}: {pair_complete} responses ({(pair_complete/total_responses*100):.1f}%)')
                print(f'{cpfq_cols[j]} - {cpfq_cols[i]}: {pair_complete} responses ({(pair_complete/total_responses*100):.1f}%)')
    
    # Summary statistics
    print('\nSummary Statistics for Pairwise Comparisons:')
    print(f'Minimum pairs: {min(n_pairs)}')
    print(f'Maximum pairs: {max(n_pairs)}')
    print(f'Average pairs: {int(np.mean(n_pairs))}')
    print(f'Median pairs: {int(np.median(n_pairs))}')
    
    # Return analysis results as string
    results = ['Missing Values Analysis', '=' * 50 + '\n']
    results.append('1. Missing Values per Item:')
    results.append(missing_summary.to_string())
    results.append('\n2. Pairwise Complete Cases Analysis:\n')
    results.append('All Pairs:')
    for info in pair_info:
        results.append(info)
    results.append('\nSummary Statistics:')
    results.append(f'Minimum pairs: {min(n_pairs)}')
    results.append(f'Maximum pairs: {max(n_pairs)}')
    results.append(f'Average pairs: {int(np.mean(n_pairs))}')
    results.append(f'Median pairs: {int(np.median(n_pairs))}')
    
    analysis_text = '\n'.join(results)
    
    return analysis_text

def print_descriptive_stats(cpfq_data: pd.DataFrame) -> str:
    """Print descriptive statistics for the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    
    Returns:
        str: The descriptive statistics as a string
    """
    # Print descriptive statistics
    desc_stats = cpfq_data.describe()
    print('\nDescriptive Statistics:')
    print(desc_stats)
    
    # Print missing values
    missing_values = cpfq_data.isnull().sum()
    print('\nMissing Values:')
    print(missing_values)
    
    # Return descriptive statistics as string
    results = ['Descriptive Statistics:']
    results.append(desc_stats.to_string())
    results.append('\nMissing Values:')
    results.append(missing_values.to_string())
    
    stats_text = '\n'.join(results)
    
    return stats_text

def plot_correlation_matrix(cpfq_data: pd.DataFrame) -> None:
    """Plot the correlation matrix for the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = get_correlation_matrix(cpfq_data)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # Create heatmap with enhanced styling
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap=CORR_PALETTE, 
                center=0,
                square=True, 
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'},
                annot_kws={'size': 8})
    
    plt.title('CPFQ Item Correlation Matrix', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', size=10)
    plt.yticks(rotation=0, size=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'))
    plt.close()

def calculate_kmo(cpfq_data: pd.DataFrame, cpfq_cols: List[str]) -> str:
    """Calculate the Kaiser-Meyer-Olkin (KMO) score for the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    
    Returns:
        str: The KMO score as a string
    """
    corr_matrix = get_correlation_matrix(cpfq_data)
    
    # Calculate KMO score
    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo_model = calculate_kmo(corr_matrix)
    
    print('\nKMO Scores:')
    print(f'Overall KMO score: {kmo_model:.3f}')
    
    print('\nKMO scores for each variable:')
    for var, score in zip(cpfq_cols, kmo_all):
        print(f'{var}: {score:.3f}')
    
    # Return KMO scores as string
    results = [f'Overall KMO score: {kmo_model:.3f}\n']
    results.append('KMO scores for each variable:')
    for var, score in zip(cpfq_cols, kmo_all):
        results.append(f'{var}: {score:.3f}')
    
    results_text = '\n'.join(results)
    return results_text

def perform_bartlett_test(cpfq_data: pd.DataFrame, cpfq_cols: List[str]) -> str:
    """Perform Bartlett's test of sphericity for the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    
    Returns:
        str: The Bartlett's test results as a string
    """
    correlation_matrix = cpfq_data.corr()
    n = len(cpfq_data)
    p = len(cpfq_cols)
    chi_square_value = -np.log(np.linalg.det(correlation_matrix)) * (n - 1 - (2 * p + 5) / 6)
    df = p * (p - 1) / 2
    p_value = stats.chi2.sf(chi_square_value, df)
    
    print('\nBartlett\'s test of sphericity:')
    print(f'Chi-square: {chi_square_value:.2f}')
    print(f'p-value: {p_value:.10f}')
    
    # Return Bartlett's test results as string
    results = ['Bartlett\'s test of sphericity:']
    results.append(f'Chi-square: {chi_square_value:.2f}')
    results.append(f'p-value: {p_value:.10f}')
    
    bartlett_text = '\n'.join(results)
    
    return bartlett_text

def create_scree_plot(cpfq_data: pd.DataFrame, cpfq_cols: List[str]) -> None:
    """Create a scree plot for the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    """
    fa = FactorAnalyzer(rotation=None, n_factors=len(cpfq_cols))
    fa.fit(cpfq_data)
    ev, v = fa.get_eigenvalues()
    
    plt.figure(figsize=(10, 6))
    
    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(ev)))
    
    # Plot bars and line
    plt.bar(range(1, len(ev) + 1), ev, alpha=0.6, color=colors)
    plt.plot(range(1, len(ev) + 1), ev, 'o-', color='#FF6B6B', linewidth=2, markersize=8)
    
    # Add Kaiser criterion line
    plt.axhline(y=1, color='#4A90E2', linestyle='--', alpha=0.8, label='Kaiser criterion')
    
    plt.title('Scree Plot of Eigenvalues', fontsize=14, pad=20)
    plt.xlabel('Factor Number', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'scree_plot.png'))
    plt.close()
    
def create_factor_importance_plot(fa: FactorAnalyzer) -> None:
    """Create a factor importance plot showing variance explained by each factor.
    
    Args:
        fa: The fitted FactorAnalyzer object
    """
    # Get variance explained
    var_exp = fa.get_factor_variance()[1][:3] * 100
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create colors
    colors = plt.cm.tab10(np.arange(3))
    
    # Create bar chart
    bars = plt.bar(range(1, 4), var_exp, color=colors, alpha=0.7)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
    
    # Add titles and labels
    plt.title('Percentage of Variance Explained by Each Factor', fontsize=14, pad=20)
    plt.xlabel('Factor Number', fontsize=12)
    plt.ylabel('Variance Explained (%)', fontsize=12)
    plt.xticks(range(1, 4))
    plt.ylim(0, max(var_exp) * 1.2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(RESULTS_DIR, 'factor_importance.png'))
    plt.close()

def get_correlation_matrix(cpfq_data: pd.DataFrame) -> pd.DataFrame:
    """Get the correlation matrix for the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    
    Returns:
        pd.DataFrame: The correlation matrix
    """
    corr_matrix = cpfq_data.corr()
    
    # Check for negative eigenvalues
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        # Add a small constant to the diagonal if needed
        corr_matrix = corr_matrix + np.eye(len(corr_matrix)) * abs(min_eig) * 1.1
    
    return corr_matrix

def perform_factor_analysis(cpfq_data: pd.DataFrame, cpfq_cols: List[str]) -> str:
    """Perform factor analysis on CPFQ data.

    Args:
        cpfq_data: DataFrame containing CPFQ data
        cpfq_cols: List of column names to include in analysis

    Returns:
        str: Results of factor analysis
    """
    try:
        corr_matrix = get_correlation_matrix(cpfq_data)
        
        # Perform factor analysis with 3 factors
        fa = FactorAnalyzer(rotation='oblimin', n_factors=3, method='principal')
        fa.fit(corr_matrix)

        # Get factor loadings
        loadings = pd.DataFrame(
            fa.loadings_,
            columns=[f'Factor{i+1}' for i in range(3)],
            index=cpfq_cols
        )

        # Calculate variance explained
        var_exp = fa.get_factor_variance()[1][:3] * 100
        cum_var_exp = np.cumsum(var_exp)
        eigenvalues = fa.get_eigenvalues()[0][:3]
        variance = pd.DataFrame(
            {
                'Eigenvalue': eigenvalues,
                'Proportion of Variance': var_exp,
                'Cumulative Variance': cum_var_exp
            }
        )

        # Format results
        results = []
        results.append('Factor Loadings:')
        results.append(loadings.to_string())
        results.append('\nVariance Explained (%):')
        results.append(variance.to_string())
        
        # Add factor interpretations
        results.append('\nFactor Interpretations:')
        results.append('===================\n')
        
        # Factor 1 - Sort and organize loadings
        f1_positive = loadings[loadings['Factor1'] > 0.3]['Factor1'].sort_values(ascending=False)
        f1_negative = loadings[loadings['Factor1'] < -0.3]['Factor1'].sort_values()
        
        # Factor 2 - Sort and organize loadings
        f2_positive = loadings[loadings['Factor2'] > 0.3]['Factor2'].sort_values(ascending=False)
        f2_negative = loadings[loadings['Factor2'] < -0.3]['Factor2'].sort_values()
        
        # Factor 3 - Sort and organize loadings
        f3_positive = loadings[loadings['Factor3'] > 0.3]['Factor3'].sort_values(ascending=False)
        f3_negative = loadings[loadings['Factor3'] < -0.3]['Factor3'].sort_values()
        
        # Factor 1
        results.append(f'Factor 1 - "Experiential Avoidance vs. Psychological Fusion" ({var_exp[0]:.2f}% of variance):')
        results.append('This factor represents the continuum between experiential avoidance (avoiding uncomfortable thoughts/feelings) and psychological fusion (overidentification with thoughts).')
        results.append('All significant loadings (|loading| > 0.3) in descending order by absolute magnitude:')
        
        # List negative loadings for Factor 1
        results.append('\nNegative loadings (experiential avoidance):')
        for item, value in f1_negative.items():
            results.append(f'- {item}: {value:.2f}')
        
        # List positive loadings for Factor 1
        results.append('\nPositive loadings (psychological fusion):')
        for item, value in f1_positive.items():
            results.append(f'- {item}: {value:.2f}')
        
        # Factor 2
        results.append(f'\n\nFactor 2 - "Committed Action" ({var_exp[1]:.2f}% of variance):')
        results.append('This factor represents persistence, effort, and goal-directed behavior aligned with personal values.')
        results.append('All significant loadings (|loading| > 0.3) in descending order by absolute magnitude:')
        
        # List positive loadings for Factor 2
        results.append('\nPositive loadings (committed action):')
        for item, value in f2_positive.items():
            results.append(f'- {item}: {value:.2f}')
        
        # List negative loadings for Factor 2
        if not f2_negative.empty:
            results.append('\nNegative loadings:')
            for item, value in f2_negative.items():
                results.append(f'- {item}: {value:.2f}')
        
        # Factor 3
        results.append(f'\n\nFactor 3 - "Present Moment Awareness and Values" ({var_exp[2]:.2f}% of variance):')
        results.append('This factor represents mindful awareness of the present moment and connection with personal values.')
        results.append('All significant loadings (|loading| > 0.3) in descending order by absolute magnitude:')
        
        # List positive loadings for Factor 3
        results.append('\nPositive loadings (awareness and values):')
        for item, value in f3_positive.items():
            results.append(f'- {item}: {value:.2f}')
        
        # List negative loadings for Factor 3
        if not f3_negative.empty:
            results.append('\nNegative loadings:')
            for item, value in f3_negative.items():
                results.append(f'- {item}: {value:.2f}')        
        
        results.append('\n')
        
        # Create visualizations
        # Factor loadings heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings, annot=True, cmap=HEATMAP_PALETTE, center=0, vmin=-1, vmax=1)
        plt.title('Factor Loadings Heatmap', fontsize=14)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('Items', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'factor_loadings_heatmap.png'))
        plt.close()
        
        # Factor importance plot showing variance explained
        create_factor_importance_plot(fa)
        
        # Join all results into a single string
        return '\n'.join(results)
    except Exception as e:
        print(f'Error in factor analysis: {str(e)}')
        return None

def save_results(results: str) -> None:
    """Save analysis results to a text file.
    
    Args:
        results: String containing all analysis results and interpretations
    """
    output_file = os.path.join(RESULTS_DIR, 'cpfq_factor_analysis_results.txt')
    try:
        with open(output_file, 'w') as f:
            f.write(results)
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        raise

def main() -> None:
    """Main function to run the factor analysis."""
    try:
        # Load data
        logging.info("Loading data...")
        df = load_data()
        
        # Prepare data
        logging.info("Preparing data...")
        cpfq_data, cpfq_cols = prepare_data(df)
        
        # Analyze missing values and pairs
        logging.info("Analyzing missing values...")
        missing_analysis = analyze_missing_and_pairs(cpfq_data, cpfq_cols)
        
        # Calculate KMO
        logging.info("Calculating KMO...")
        kmo_analysis = calculate_kmo(cpfq_data, cpfq_cols)
        
        # Perform Bartlett's test
        logging.info("Performing Bartlett's test...")
        bartlett_analysis = perform_bartlett_test(cpfq_data, cpfq_cols)
        
        # Create visualizations
        logging.info("Creating visualizations...")
        plot_correlation_matrix(cpfq_data)
        create_scree_plot(cpfq_data, cpfq_cols)
        
        # Perform factor analysis
        logging.info("Performing factor analysis...")
        fa_results = perform_factor_analysis(cpfq_data, cpfq_cols)
        
        # Combine all results
        results = (
            "CPFQ Factor Analysis Results\n"
            "===========================\n\n"
            f"Generated on: {datetime.datetime.now()}\n\n"
            f"{missing_analysis}\n\n"
            f"{kmo_analysis}\n\n"
            f"{bartlett_analysis}\n\n"
            f"{fa_results}\n"
        )
        
        # Save results
        logging.info("Saving results...")
        save_results(results)
        
        # Print summary
        print("\nAnalysis complete!")
        print(f"Results have been saved to the '{RESULTS_DIR}' directory:")
        print(f"1. {os.path.join(RESULTS_DIR, 'cpfq_factor_analysis_results.txt')}")
        print(f"2. {os.path.join(RESULTS_DIR, 'correlation_matrix.png')}")
        print(f"3. {os.path.join(RESULTS_DIR, 'scree_plot.png')}")
        print(f"4. {os.path.join(RESULTS_DIR, 'factor_correlations.png')}")
        print(f"5. {os.path.join(RESULTS_DIR, 'factor_importance.png')}")
        print(f"6. {os.path.join(RESULTS_DIR, 'factor_loadings_heatmap.png')}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
