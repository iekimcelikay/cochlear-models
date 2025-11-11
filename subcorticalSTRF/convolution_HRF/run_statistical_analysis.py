#!/usr/bin/env python3
"""
Usage Script for BEZ Statistical Testing and Plot Generation
Created: 2025-10-27

This script demonstrates how to run the statistical testing analysis
on BEZ convolved responses and generate various plots for visualization.

Usage Examples:
1. Basic analysis with plots displayed:
   python run_statistical_analysis.py

2. Save plots to files:
   python run_statistical_analysis.py --save_plots

3. Analyze specific fiber types:
   python run_statistical_analysis.py --fiber_types hsr msr

4. Quick test with limited conditions:
   python run_statistical_analysis.py --max_conditions 3

5. Full analysis with all options:
   python run_statistical_analysis.py --save_plots --fiber_types hsr msr lsr --output_dir ./my_results
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# Add the convolution_HRF directory to Python path
script_dir = Path(__file__).parent
convolution_dir = script_dir / "subcorticalSTRF" / "convolution_HRF"
sys.path.append(str(convolution_dir))

def find_data_files(base_dir="."):
    """
    Find available convolved response data files in the workspace.
    
    Returns:
    - data_files: List of paths to pickle files containing convolved responses
    """
    data_files = []
    
    # Common locations for convolved response files
    search_patterns = [
        "**/convolved_responses*.pkl",
        "**/bez_convolved*.pkl", 
        "**/convolution_results*.pkl",
        "**/*convolved*.pkl"
    ]
    
    for pattern in search_patterns:
        files = list(Path(base_dir).glob(pattern))
        data_files.extend(files)
    
    # Remove duplicates and sort
    data_files = sorted(list(set(data_files)))
    
    return data_files


def check_dependencies():
    """Check if required modules are available."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import scipy.stats
        import pandas as pd
        print("✓ All required dependencies are available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install numpy matplotlib seaborn scipy pandas")
        return False


def run_basic_analysis(data_file, output_dir="./statistical_results", 
                      fiber_types=None, max_conditions=None, save_plots=False):
    """
    Run basic statistical analysis with default parameters.
    
    Parameters:
    - data_file: Path to convolved responses pickle file
    - output_dir: Directory to save results
    - fiber_types: List of fiber types to analyze (default: ['hsr'])
    - max_conditions: Maximum conditions to analyze (for testing)
    - save_plots: Whether to save plots to files
    """
    if fiber_types is None:
        fiber_types = ['hsr']
    
    # Build command
    cmd = [
        sys.executable, 
        str(convolution_dir / "statistical_testing_bez_runs.py"),
        "--data_file", str(data_file),
        "--output_dir", str(output_dir),
        "--fiber_types"] + fiber_types
    
    if max_conditions:
        cmd.extend(["--max_conditions", str(max_conditions)])
    
    if save_plots:
        cmd.append("--save_plots")
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"This will analyze {', '.join(fiber_types)} fiber types and display plots...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✓ Analysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Analysis failed with error: {e}")
        return False


def run_comprehensive_analysis(data_file, output_dir="./comprehensive_results"):
    """
    Run comprehensive analysis with all fiber types and saved plots.
    
    Parameters:
    - data_file: Path to convolved responses pickle file
    - output_dir: Directory to save results
    """
    cmd = [
        sys.executable,
        str(convolution_dir / "statistical_testing_bez_runs.py"),
        "--data_file", str(data_file),
        "--output_dir", str(output_dir),
        "--fiber_types", "hsr", "msr", "lsr",
        "--save_plots",
        "--correlation_type", "pearson"
    ]
    
    print("Running comprehensive analysis...")
    print("This will:")
    print("- Analyze all fiber types (HSR, MSR, LSR)")
    print("- Generate and save all plots")
    print("- Create summary statistics")
    print("- Save results to files")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Comprehensive analysis completed! Results saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Analysis failed with error: {e}")
        return False


def run_quick_test(data_file, output_dir="./test_results"):
    """
    Run quick test analysis with limited conditions for demonstration.
    
    Parameters:
    - data_file: Path to convolved responses pickle file  
    - output_dir: Directory to save results
    """
    cmd = [
        sys.executable,
        str(convolution_dir / "statistical_testing_bez_runs.py"),
        "--data_file", str(data_file),
        "--output_dir", str(output_dir),
        "--fiber_types", "hsr",
        "--max_conditions", "3",
        "--save_plots"
    ]
    
    print("Running quick test analysis...")
    print("This will analyze only the first 3 conditions for demonstration.")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Quick test completed! Results saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Test failed with error: {e}")
        return False


def display_analysis_menu():
    """Display interactive menu for analysis options."""
    print("\n" + "="*60)
    print("BEZ Statistical Analysis - Interactive Menu")
    print("="*60)
    print("1. Quick test (3 conditions, HSR only)")
    print("2. Basic analysis (HSR fibers, plots displayed)")
    print("3. Multi-fiber analysis (HSR + MSR, plots displayed)")
    print("4. Comprehensive analysis (All fibers, plots saved)")
    print("5. Custom analysis (specify parameters)")
    print("6. List available data files")
    print("7. Exit")
    print("="*60)


def interactive_mode():
    """Run interactive mode for selecting analysis options."""
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Find available data files
    data_files = find_data_files()
    
    if not data_files:
        print("No convolved response data files found!")
        print("Please ensure you have run the convolution analysis first.")
        return
    
    print(f"Found {len(data_files)} data file(s):")
    for i, file in enumerate(data_files, 1):
        print(f"  {i}. {file}")
    
    # Select data file
    while True:
        try:
            file_choice = input(f"\nSelect data file (1-{len(data_files)}): ")
            file_idx = int(file_choice) - 1
            if 0 <= file_idx < len(data_files):
                selected_file = data_files[file_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            return
    
    print(f"Selected: {selected_file}")
    
    # Analysis menu loop
    while True:
        display_analysis_menu()
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                print("\n--- Quick Test Analysis ---")
                run_quick_test(selected_file)
                
            elif choice == '2':
                print("\n--- Basic Analysis ---")
                run_basic_analysis(selected_file, fiber_types=['hsr'])
                
            elif choice == '3':
                print("\n--- Multi-Fiber Analysis ---")
                run_basic_analysis(selected_file, fiber_types=['hsr', 'msr'])
                
            elif choice == '4':
                print("\n--- Comprehensive Analysis ---")
                run_comprehensive_analysis(selected_file)
                
            elif choice == '5':
                print("\n--- Custom Analysis ---")
                print("Available fiber types: hsr, msr, lsr")
                fiber_input = input("Enter fiber types (space-separated, e.g., 'hsr msr'): ")
                fiber_types = fiber_input.split()
                
                save_choice = input("Save plots to files? (y/n): ").lower().startswith('y')
                
                max_cond_input = input("Max conditions (press Enter for all): ").strip()
                max_conditions = int(max_cond_input) if max_cond_input else None
                
                output_dir = input("Output directory (press Enter for default): ").strip()
                if not output_dir:
                    output_dir = "./custom_results"
                
                run_basic_analysis(
                    selected_file, 
                    output_dir=output_dir,
                    fiber_types=fiber_types,
                    max_conditions=max_conditions,
                    save_plots=save_choice
                )
                
            elif choice == '6':
                print("\n--- Available Data Files ---")
                for i, file in enumerate(data_files, 1):
                    print(f"  {i}. {file}")
                    
            elif choice == '7':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function with command line argument handling."""
    parser = argparse.ArgumentParser(
        description='Usage script for BEZ statistical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--data_file', type=str,
                       help='Path to convolved responses pickle file')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with 3 conditions')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive analysis with all fiber types')
    parser.add_argument('--fiber_types', nargs='+', default=['hsr'],
                       help='Fiber types to analyze (default: hsr)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--output_dir', type=str, default='./statistical_results',
                       help='Output directory for results')
    parser.add_argument('--max_conditions', type=int,
                       help='Maximum number of conditions to analyze')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # If no arguments provided or interactive mode requested
    if len(sys.argv) == 1 or args.interactive:
        interactive_mode()
        return
    
    # Find data file if not provided
    if not args.data_file:
        data_files = find_data_files()
        if not data_files:
            print("No convolved response data files found!")
            print("Please specify --data_file or ensure data files exist.")
            return
        args.data_file = data_files[0]
        print(f"Using data file: {args.data_file}")
    
    # Run specific analysis based on flags
    if args.quick_test:
        run_quick_test(args.data_file, args.output_dir)
    elif args.comprehensive:
        run_comprehensive_analysis(args.data_file, args.output_dir)
    else:
        run_basic_analysis(
            args.data_file,
            output_dir=args.output_dir,
            fiber_types=args.fiber_types,
            max_conditions=args.max_conditions,
            save_plots=args.save_plots
        )


if __name__ == "__main__":
    main()
