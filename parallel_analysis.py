"""
Parallel Analysis for Factor Analysis

This script implements parallel analysis to determine the optimal number of factors
to retain in factor analysis for the CPFQ dataset. The method compares eigenvalues 
from the actual data against eigenvalues from randomly generated data with the same 
dimensions.

The analysis includes:
- Generation of random datasets
- Comparison of actual vs random eigenvalues
- Visualization of parallel analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import logging
from typing import List, Tuple
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Import existing functions from main analysis file
from cpfq_factor_analysis import load_data, prepare_data, RESULTS_DIR

# Configuration
ITERATIONS = 1000  # Number of random datasets to generate
PERCENTILE = 95    # Percentile of random eigenvalues to use
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Plot settings
sns.set_style('whitegrid')
sns.set_context('paper')
PLOT_PALETTE = 'viridis'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_random_eigenvalues(data: pd.DataFrame, n_iterations: int = ITERATIONS, 
                               percentile: int = PERCENTILE) -> np.ndarray:
    """
    Generate eigenvalues from random data with same dimensions as input data.
    
    Args:
        data: The original data
        n_iterations: Number of random datasets to generate
        percentile: Percentile of random eigenvalues to use
    
    Returns:
        np.ndarray: Array of eigenvalues at specified percentile
    """
    n_samples, n_vars = data.shape
    logging.info(f"Generating {n_iterations} random datasets with shape {n_samples}x{n_vars}")
    
    # Store eigenvalues from each iteration
    random_evals = np.zeros((n_iterations, n_vars))
    
    for i in range(n_iterations):
        # Generate random normal data with same dimensions
        random_data = np.random.normal(0, 1, size=(n_samples, n_vars))
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(random_data, rowvar=False)
        # Calculate eigenvalues
        evals, _ = np.linalg.eig(corr_matrix)
        # Sort eigenvalues in descending order
        evals = np.sort(evals)[::-1]
        random_evals[i, :] = evals
        
        # Log progress occasionally
        if (i + 1) % 100 == 0 or i == 0:
            logging.info(f"Completed {i + 1}/{n_iterations} iterations")
    
    # Calculate the percentile across iterations
    percentile_evals = np.percentile(random_evals, percentile, axis=0)
    return percentile_evals


def get_actual_eigenvalues(data: pd.DataFrame) -> np.ndarray:
    """
    Calculate eigenvalues from the actual data correlation matrix.
    
    Args:
        data: The original data
    
    Returns:
        np.ndarray: Array of eigenvalues in descending order
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    # Calculate eigenvalues
    evals, _ = np.linalg.eig(corr_matrix.values)
    # Sort eigenvalues in descending order
    evals = np.sort(evals)[::-1]
    return evals


def plot_parallel_analysis(actual_evals: np.ndarray, random_evals: np.ndarray, 
                          output_dir: str = RESULTS_DIR) -> None:
    """
    Create a plot comparing actual eigenvalues to random eigenvalues.
    
    Args:
        actual_evals: Eigenvalues from actual data
        random_evals: Eigenvalues from random data
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    n_factors = len(actual_evals)
    factor_numbers = np.arange(1, n_factors + 1)
    
    # Plot actual eigenvalues
    plt.plot(factor_numbers, actual_evals, 'o-', color='blue', 
             label=f'Actual Data Eigenvalues')
    
    # Add line at eigenvalue = 1
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Eigenvalue = 1')
    
    # Plot random eigenvalues
    plt.plot(factor_numbers, random_evals, 's--', color='red', 
             label=f'{PERCENTILE}th Percentile of Random Eigenvalues')
    
    # Count factors where actual eigenvalues > 1
    factors_to_retain = sum(actual_evals > 1)
    
    # Draw a vertical line at the suggested cut-off
    if factors_to_retain > 0:
        plt.axvline(x=factors_to_retain + 0.5, color='green', linestyle='-', alpha=0.5)
        plt.text(factors_to_retain + 0.6, min(actual_evals) * 1.1, 
                f'Suggested factors: {factors_to_retain}', color='green')
    
    # Customize plot
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.title('Parallel Analysis Scree Plot')
    plt.xticks(factor_numbers)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'parallel_analysis_scree_plot.{FIGURE_FORMAT}')
    plt.savefig(output_file, dpi=FIGURE_DPI)
    plt.close()
    logging.info(f"Saved parallel analysis plot to {output_file}")
    
    return factors_to_retain


def perform_parallel_analysis(data: pd.DataFrame) -> Tuple[int, str]:
    """
    Perform parallel analysis and return suggested number of factors.
    
    Args:
        data: The prepared data for analysis
    
    Returns:
        Tuple containing:
            - int: Suggested number of factors to retain
            - str: Analysis results as a string
    """
    logging.info("Starting parallel analysis...")
    
    # Calculate actual eigenvalues
    actual_evals = get_actual_eigenvalues(data)
    
    # Generate random eigenvalues
    random_evals = generate_random_eigenvalues(data, n_iterations=ITERATIONS, percentile=PERCENTILE)
    
    # Plot results
    factors_to_retain = plot_parallel_analysis(actual_evals, random_evals)
    
    # Prepare results string
    results = []
    results.append("Parallel Analysis Results")
    results.append("=======================\n")
    
    results.append(f"Number of iterations: {ITERATIONS}")
    results.append(f"Percentile used: {PERCENTILE}th\n")
    
    results.append("Eigenvalues Comparison:")
    results.append("-----------------------")
    results.append(f"{'Factor':<8}{'Actual EV':<15}{'Retain':<10}")
    results.append("-" * 35)
    
    for i, actual in enumerate(actual_evals):
        retain = "Yes" if actual > 1 else "No"
        results.append(f"{i+1:<8}{actual:<15.4f}{retain:<10}")
    
    results.append("\nSuggested number of factors to retain: " + str(factors_to_retain))
    
    if factors_to_retain == 0:
        results.append("\nWARNING: No factors meet the retention criteria. Consider using a lower percentile.")
    
    return factors_to_retain, "\n".join(results)


def save_results(results: str) -> None:
    """
    Save analysis results to a text file.
    
    Args:
        results: String containing all analysis results
    """
    output_file = os.path.join(RESULTS_DIR, 'parallel_analysis_results.txt')
    try:
        with open(output_file, 'w') as f:
            f.write(results)
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        raise


def main() -> None:
    """Main function to run the parallel analysis."""
    try:
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Load data
        logging.info("Loading data...")
        df = load_data()
        
        # Prepare data
        logging.info("Preparing data...")
        cpfq_data, cpfq_cols = prepare_data(df)
        
        # Perform parallel analysis
        logging.info("Performing parallel analysis...")
        suggested_factors, results_text = perform_parallel_analysis(cpfq_data)
        
        # Save results
        logging.info("Saving results...")
        save_results(results_text)
        
        # Print summary
        print("\nParallel Analysis complete!")
        print(f"Suggested number of factors to retain: {suggested_factors}")
        print(f"Results have been saved to the '{RESULTS_DIR}' directory:")
        print(f"1. {os.path.join(RESULTS_DIR, 'parallel_analysis_results.txt')}")
        print(f"2. {os.path.join(RESULTS_DIR, 'parallel_analysis_scree_plot.png')}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
