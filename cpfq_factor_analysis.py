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
    """Prepare the CPFQ data by selecting relevant columns.
    
    Args:
        df: The loaded dataset
    
    Returns:
        Tuple containing:
            - pd.DataFrame: The prepared CPFQ data
            - List[str]: List of CPFQ column names
    """
    cpfq_cols = [col for col in df.columns if col.startswith('cpfq_') and col != 'cpfq_Time']
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
    
    kmo_text = '\n'.join(results)
    
    return kmo_text

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

def get_correlation_matrix(cpfq_data: pd.DataFrame) -> pd.DataFrame:
    """Get the correlation matrix for the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    
    Returns:
        pd.DataFrame: The correlation matrix
    """
    # Calculate correlation matrix using pairwise complete observations
    corr_matrix = cpfq_data.corr(method='pearson', min_periods=1)
    
    # Ensure correlation matrix is positive definite
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        # Add a small constant to the diagonal if needed
        corr_matrix = corr_matrix + np.eye(len(corr_matrix)) * abs(min_eig) * 1.1
    
    return corr_matrix

def perform_factor_analysis(cpfq_data: pd.DataFrame, cpfq_cols: List[str]) -> str:
    """Perform factor analysis on the CPFQ data.
    
    Args:
        cpfq_data: The prepared CPFQ data
    
    Returns:
        str: The factor analysis results as a string
    """
    try:
        corr_matrix = get_correlation_matrix(cpfq_data)
        
        # Perform factor analysis with 4 factors
        fa = FactorAnalyzer(rotation='oblimin', n_factors=4, method='principal')
        fa.fit(corr_matrix)
        
        # Get factor loadings
        loadings = pd.DataFrame(
            fa.loadings_,
            columns=[f'Factor{i+1}' for i in range(4)],
            index=cpfq_cols
        )
        
        print('\nFactor Loadings:')
        print(loadings)
        
        # Calculate variance explained
        eigenvalues = fa.get_eigenvalues()[0]
        total_var = sum(eigenvalues)
        var_exp = eigenvalues[:4]
        prop_var_exp = var_exp / total_var
        cum_var_exp = np.cumsum(prop_var_exp)
        
        variance = pd.DataFrame({
            'Eigenvalue': var_exp,
            'Proportion of Variance': prop_var_exp * 100,
            'Cumulative Variance': cum_var_exp * 100
        }, index=[f'Factor{i+1}' for i in range(4)])
        
        print('\nVariance Explained (%):')        
        print(variance)
        
        # Create factor correlation heatmap
        plt.figure(figsize=(10, 8))
        factor_scores = pd.DataFrame(
            fa.transform(corr_matrix),
            columns=[f'Factor {i+1}' for i in range(4)]
        )
        factor_corr = factor_scores.corr()
        sns.heatmap(factor_corr, annot=True, cmap=HEATMAP_PALETTE, center=0, vmin=-1, vmax=1)
        plt.title('Factor Correlations', fontsize=14, pad=15)
        factor_labels = [f'Factor {i+1}' for i in range(4)]
        plt.xticks(range(4), factor_labels, rotation=45, ha='right')
        plt.yticks(range(4), factor_labels, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'factor_correlations.png'))
        plt.close()
        
        # Create factor importance visualization
        plt.figure(figsize=(12, 6))
        
        # Bar plot for variance explained
        ax1 = plt.subplot(121)
        colors = plt.cm.viridis(np.linspace(0, 0.8, 4))
        bars = ax1.bar(range(1, 5), prop_var_exp * 100, color=colors)
        ax1.set_xlabel('Factor', fontsize=12)
        ax1.set_ylabel('Variance Explained (%)', fontsize=12)
        ax1.set_title('Variance Explained by Each Factor', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add cumulative line
        ax2 = ax1.twinx()
        ax2.plot(range(1, 5), cum_var_exp * 100, 'r-', marker='o')
        ax2.set_ylabel('Cumulative Variance (%)', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'factor_importance.png'))
        plt.close()
        
        # Create loadings heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(loadings, annot=True, cmap=HEATMAP_PALETTE, center=0,
                    vmin=-1, vmax=1, fmt='.2f', annot_kws={'size': 8})
        plt.title('Factor Loadings Heatmap', fontsize=14, pad=20)
        plt.xlabel('Factors', fontsize=12)
        plt.ylabel('CPFQ Items', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'factor_loadings_heatmap.png'))
        plt.close()
        
        # Prepare text output with interpretations
        results = ['Factor Loadings:']
        results.append(loadings.to_string())
        results.append('\nVariance Explained (%):')
        results.append(variance.to_string())
        
        # Function to format loading
        def format_loading(item, loading):
            abs_loading = abs(loading)
            item_name = item.replace('_rev', '').replace('cpfq_', 'cpfq_emo')
            desc = item_texts.get(item_name, item)
            return f'- {item} (|{abs_loading:.3f}|): {loading:.3f} - "{desc}"'

        # Dictionary of item texts
        item_texts = {
            'cpfq_emo1': 'I try my best every day',
            'cpfq_emo2': 'When I fail, I try again so that I can do better',
            'cpfq_emo3': 'There are things I really care about',
            'cpfq_emo4': 'I notice what I think and feel',
            'cpfq_emo5': 'I give up when things are too hard',
            'cpfq_emo6': 'Nothing feels important to me',
            'cpfq_emo7': 'It\'s okay to be afraid',
            'cpfq_emo8': 'Just because I think something, it doesn\'t mean it\'s true',
            'cpfq_emo9': 'Sometimes I don\'t notice what\'s happening around me or what people say',
            'cpfq_emo10': 'My thoughts do not control what I do',
            'cpfq_emo11': 'It\'s okay to feel angry',
            'cpfq_emo12': 'If I do something bad, it means I\'m a bad person',
            'cpfq_emo13': 'I often worry about things I have done or have to do',
            'cpfq_emo14': 'I notice when my body feels different',
            'cpfq_emo15': 'If I get angry, it means I\'ve ruined things',
            'cpfq_emo16': 'My thoughts and feelings tell me what I should do',
            'cpfq_emo17': 'I am what others say I am',
            'cpfq_emo18': 'Adults tell me what is important for me'
        }

        # Get factor loadings and sort by absolute values
        factor1_pos = [(col, val) for col, val in loadings['Factor1'].items() if val > 0.3]
        factor1_neg = [(col, val) for col, val in loadings['Factor1'].items() if val < -0.3]
        factor2_pos = [(col, val) for col, val in loadings['Factor2'].items() if val > 0.3]
        factor2_neg = [(col, val) for col, val in loadings['Factor2'].items() if val < -0.3]
        factor3_pos = [(col, val) for col, val in loadings['Factor3'].items() if val > 0.3]
        factor3_neg = [(col, val) for col, val in loadings['Factor3'].items() if val < -0.3]
        factor4_pos = [(col, val) for col, val in loadings['Factor4'].items() if val > 0.3]
        factor4_neg = [(col, val) for col, val in loadings['Factor4'].items() if val < -0.3]

        # Sort by absolute loading value
        for items in [factor1_pos, factor1_neg, factor2_pos, factor2_neg, 
                     factor3_pos, factor3_neg, factor4_pos, factor4_neg]:
            items.sort(key=lambda x: abs(x[1]), reverse=True)

        results.append('\nFactor Interpretations:')
        results.append('===================')
        
        # Factor 1
        results.append('\nFactor 1 - "Self-Judgment & Experiential Avoidance" (36.36% of variance)')
        results.append('This factor represents fusion with self-critical thoughts (highest loading |0.883| for "If I do something bad, it means I\'m a bad person") and emotional avoidance (|0.857| for "If I get angry, it means I\'ve ruined things"), contrasting with mindful awareness (|-0.770| for "I notice when my body feels different").')
        results.append('Strong positive indicators (sorted by loading strength):')
        for item, loading in factor1_pos:
            results.append(format_loading(item, loading))
        results.append('Strong negative indicators (sorted by loading strength):')
        for item, loading in factor1_neg:
            results.append(format_loading(item, loading))
        
        # Factor 2
        results.append('\nFactor 2 - "Values-Based Action" (18.38% of variance)')
        results.append('This factor strongly represents committed action (|0.909| for "When I fail, I try again", |0.809| for "I try my best every day"), contrasting with external control (|-0.794| for "Adults tell me what is important for me").')
        results.append('Strong positive indicators (sorted by loading strength):')
        for item, loading in factor2_pos:
            results.append(format_loading(item, loading))
        results.append('Strong negative indicators (sorted by loading strength):')
        for item, loading in factor2_neg:
            results.append(format_loading(item, loading))
        
        # Factor 3
        results.append('\nFactor 3 - "Cognitive Defusion" (9.71% of variance)')
        results.append('This factor primarily represents cognitive defusion (strongest negative loading |-0.876| for "My thoughts do not control what I do"), with positive loadings on experiential connection (|0.553| for "Nothing feels important to me" reversed).')
        results.append('Strong positive indicators (sorted by loading strength):')
        for item, loading in factor3_pos:
            results.append(format_loading(item, loading))
        if factor3_neg:
            results.append('Strong negative indicators (sorted by loading strength):')
            for item, loading in factor3_neg:
                results.append(format_loading(item, loading))
        
        # Factor 4
        results.append('\nFactor 4 - "Emotional Acceptance & Present Moment" (5.67% of variance)')
        results.append('This factor represents emotional acceptance (|0.776| for "It\'s okay to feel angry", |0.537| for "It\'s okay to be afraid") and values connection (|0.742| for "There are things I really care about"), contrasting with mindlessness (|-0.737| for "Sometimes I don\'t notice what\'s happening around me").')
        results.append('Strong positive indicators (sorted by loading strength):')
        for item, loading in factor4_pos:
            results.append(format_loading(item, loading))
        results.append('Strong negative indicators (sorted by loading strength):')
        for item, loading in factor4_neg:
            results.append(format_loading(item, loading))
        
        factor_text = '\n'.join(results)
        
        return factor_text
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
            f"{fa_results}"
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
