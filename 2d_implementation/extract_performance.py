#!/usr/bin/env python3
"""
Script per estrarre i tempi di esecuzione dai log e generare plot di speedup ed efficiency
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
DATASETS = ["bodies_2d_10.csv", "bodies_2d_100.csv", "bodies_2d_500.csv", "bodies_2d_1000.csv", "bodies_2d_2000.csv", "bodies_2d_3000.csv", 
            "bodies_2d_5000.csv", "bodies_2d_10000.csv"]#, "bodies_2d_20000.csv", "bodies_2d_30000.csv"]
PROCESSORS = [2, 3, 4, 5, 6, 7, 8]

def extract_execution_time(log_file):
    """Extracts execution time from a log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Pattern for sequential execution time
        seq_pattern = r"Execution time for sequential:\s+([0-9.]+)\s+seconds"
        # Pattern for parallel runtime
        par_pattern = r"Execution time for parallel:\s+([0-9.]+)\s+seconds"
        
        # Try the parallel pattern first, then the sequential pattern
        match = re.search(par_pattern, content)
        if not match:
            match = re.search(seq_pattern, content)
        
        if match:
            return float(match.group(1))
        else:
            print(f"Warning: No execution time found in {log_file}")
            return None
            
    except FileNotFoundError:
        print(f"Warning: File {log_file} not found")
        return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None

def collect_performance_data():
    """Collects all performance data from log files"""
    data = []

    SEQ_DIR = os.path.join(RESULTS_DIR, "seq_dataset_2d")
    PAR_DIR = os.path.join(RESULTS_DIR, "par_dataset_2d")

    # Sequential data collection
    for dataset in DATASETS:
        seq_log = os.path.join(SEQ_DIR, f"{dataset}.log")
        seq_time = extract_execution_time(seq_log)
        
        if seq_time is not None:
            data.append({
                'dataset_2d': dataset.replace('.csv', ''),
                'processors': 1,
                'time': seq_time,
                'type': 'sequential'
            })
    
    # Parallel data collection
    for dataset in DATASETS:
        for proc in PROCESSORS:
            par_log = os.path.join(PAR_DIR, f"{dataset}_p{proc}.log")
            par_time = extract_execution_time(par_log)
            
            if par_time is not None:
                data.append({
                    'dataset_2d': dataset.replace('.csv', ''),
                    'processors': proc,
                    'time': par_time,
                    'type': 'parallel'
                })
    
    return pd.DataFrame(data)

def calculate_metrics(df):
    """Calculates speedup and efficiency"""
    results = []
    
    for dataset in df['dataset_2d'].unique():
        dataset_data = df[df['dataset_2d'] == dataset]
        
        # Find the sequential time
        seq_data = dataset_data[dataset_data['processors'] == 1]
        if seq_data.empty:
            print(f"Warning: No sequential data for {dataset}")
            continue
            
        seq_time = seq_data['time'].iloc[0]
        
        # Calculates metrics for each parallel configuration
        par_data = dataset_data[dataset_data['processors'] > 1]
        
        for _, row in par_data.iterrows():
            speedup = seq_time / row['time']
            efficiency = speedup / row['processors']
            results.append({
                'dataset_2d': dataset,
                'processors': row['processors'],
                'seq_time': seq_time,
                'par_time': row['time'],
                'speedup': speedup,
                'efficiency': efficiency
            })
    
    return pd.DataFrame(results)

def create_plots(metrics_df):
    """Create speedup and efficiency plots"""
    if metrics_df.empty:
        print("No data available for plotting")
        return
    
    Path(PLOTS_DIR).mkdir(exist_ok=True)
    
    datasets = metrics_df['dataset_2d'].unique()
    n_datasets = len(datasets)
    
    # Plot Speedup
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('N-Body Simulation Performance Analysis', fontsize=16, fontweight='bold')
    
    # Speedup plot
    ax1 = plt.subplot(2, 2, 1)
    for dataset in datasets:
        data = metrics_df[metrics_df['dataset_2d'] == dataset]
        processors = data['processors'].to_numpy()
        speedup = data['speedup'].to_numpy()
        ax1.plot(processors, speedup, 'o-', label=dataset, linewidth=2, markersize=6)
    
    # Ideal theoretical line
    max_proc = metrics_df['processors'].max()
    ideal_proc = range(2, max_proc + 1)
    ax1.plot(ideal_proc, ideal_proc, 'k--', alpha=0.7, label='Ideal Speedup')
    
    ax1.set_xlabel('Number of Processors')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Number of Processors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(2, max_proc + 1))
    
    # Efficiency plot
    ax2 = plt.subplot(2, 2, 2)
    for dataset in datasets:
        data = metrics_df[metrics_df['dataset_2d'] == dataset]
        processors = data['processors'].to_numpy()
        speedup = data['speedup'].to_numpy()
        ax1.plot(processors, speedup, 'o-', label=dataset, linewidth=2, markersize=6)
    
    # Ideal line efficiency = 1
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal Efficiency')
    
    ax2.set_xlabel('Number of Processors')
    ax2.set_ylabel('Efficiency')
    ax2.set_title('Efficiency vs Number of Processors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(2, max_proc + 1))
    ax2.set_ylim(0, 1.1)
    
    # Execution time comparison
    ax3 = plt.subplot(2, 2, 3)
    x_pos = np.arange(len(datasets))
    seq_times = [metrics_df[metrics_df['dataset_2d'] == d]['seq_time'].iloc[0] for d in datasets]
    
    # Parallel times for 8 processors
    par_times_8 = []
    for dataset in datasets:
        data = metrics_df[(metrics_df['dataset_2d'] == dataset) & (metrics_df['processors'] == 8)]
        if not data.empty:
            par_times_8.append(data['par_time'].iloc[0])
        else:
            # Get the time with the most processors available
            data = metrics_df[metrics_df['dataset_2d'] == dataset]
            max_proc_data = data[data['processors'] == data['processors'].max()]
            par_times_8.append(max_proc_data['par_time'].iloc[0])
    
    width = 0.35
    ax3.bar(x_pos - width/2, seq_times, width, label='Sequential', alpha=0.8)
    ax3.bar(x_pos + width/2, par_times_8, width, label='Parallel (8 proc)', alpha=0.8)
    
    ax3.set_xlabel('Dataset_2d')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Sequential vs Parallel Execution Time')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(datasets, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Logarithmic scale for very different times
    
    # Scalability analysis
    ax4 = plt.subplot(2, 2, 4)
    for dataset in datasets:
        data = metrics_df[metrics_df['dataset_2d'] == dataset]
        processors = np.array(data['processors'])
        speedup = np.array(data['speedup'])
        speedup_ratio = speedup / processors
        ax4.plot(processors, speedup_ratio, 'o-', label=dataset, linewidth=2, markersize=6)
    
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Perfect Scaling')
    ax4.set_xlabel('Number of Processors')
    ax4.set_ylabel('Normalized Speedup')
    ax4.set_title('Scalability Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(2, max_proc + 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'nbody_performance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also saves individual plots for each metric
    create_individual_plots(metrics_df)

def create_individual_plots(metrics_df):
    """Create individual plots for speedup and efficiency"""
    datasets = metrics_df['dataset_2d'].unique()
    
    # Individual Plot Speedup
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        data = metrics_df[metrics_df['dataset_2d'] == dataset]
        processors = data['processors'].to_numpy()
        speedup = data['speedup'].to_numpy()
        plt.plot(processors, speedup, 'o-', label=dataset, linewidth=2, markersize=8)

    
    # Ideal line
    max_proc = metrics_df['processors'].max()
    ideal_proc = range(2, max_proc + 1)
    plt.plot(ideal_proc, ideal_proc, 'k--', alpha=0.7, label='Ideal Speedup', linewidth=2)
    
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('N-Body Simulation: Speedup Analysis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(2, max_proc + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'speedup_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Individual Plot Efficiency
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        data = metrics_df[metrics_df['dataset_2d'] == dataset]
        processors = data['processors'].to_numpy()
        efficiency = data['efficiency'].to_numpy()
        plt.plot(processors, efficiency, 'o-', label=dataset, linewidth=2, markersize=8)
    
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal Efficiency', linewidth=2)
    
    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title('N-Body Simulation: Efficiency Analysis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(2, max_proc + 1))
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(metrics_df):
    """Print a summary table of the results"""
    if metrics_df.empty:
        print("No data available for summary")
        return
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    for dataset in metrics_df['dataset_2d'].unique():
        data = metrics_df[metrics_df['dataset_2d'] == dataset]
        print(f"\nDataset_2d: {dataset}")
        print("-" * 50)
        print(f"{'Proc':<5} {'Seq Time':<10} {'Par Time':<10} {'Speedup':<10} {'Efficiency':<10}")
        print("-" * 50)
        
        seq_time = data['seq_time'].iloc[0]
        print(f"{1:<5} {seq_time:<10.4f} {'-':<10} {'-':<10} {'-':<10}")
        
        for _, row in data.iterrows():
            print(f"{row['processors']:<5} {seq_time:<10.4f} {row['par_time']:<10.4f} "
                  f"{row['speedup']:<10.2f} {row['efficiency']:<10.3f}")

def main():
    print("N-Body Performance Analysis")
    print("=" * 40)
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: {RESULTS_DIR} directory not found. Run 'make test' first.")
        return
    
    # Data Collection
    print("Collecting performance data...")
    df = collect_performance_data()
    
    if df.empty:
        print("No performance data found. Make sure tests have been run successfully.")
        return
    
    print(f"Found {len(df)} data points")
    
    # Metric calculation
    print("Calculating speedup and efficiency...")
    metrics_df = calculate_metrics(df)
    
    if metrics_df.empty:
        print("Could not calculate metrics. Check that sequential and parallel data are available.")
        return
    
    # Print summary
    print_summary_table(metrics_df)
    
    # Save data to CSV
    metrics_df.to_csv(os.path.join(PLOTS_DIR, 'performance_metrics.csv'), index=False)
    print(f"\nPerformance data saved to {PLOTS_DIR}/performance_metrics.csv")
    
    # Plot generation
    print("Generating performance plots...")
    create_plots(metrics_df)
    
    print(f"\nPlots saved to {PLOTS_DIR}/")
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()