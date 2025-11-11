# Statistical Testing Script Usage Examples

This document provides usage examples for the `statistical_testing_bez_runs.py` script for analyzing BEZ run correlations and comparing with cochlea responses.

## Overview

The script performs statistical analysis on BEZ runs convolved with HRF to assess:
- Consistency of responses across runs using pairwise correlations
- Reliability of individual runs
- Statistical significance of correlations
- Comparison with cochlea responses

## Basic Usage

### 1. Single Condition Analysis
Analyze a specific condition (fiber type, CF, frequency, and dB level) to see the pairwise correlation matrix:

```bash
python /home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/convolution_HRF/statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --analyze_single_condition \
  --fiber_type hsr \
  --cf_val 1000 \
  --freq_val 1000 \
  --db_val 60
```

**Output**: Displays pairwise correlation matrix, distribution plots, and statistical summary for the specified condition.

### 2. Cochlea-BEZ Correlation Analysis
Compare cochlea responses with BEZ mean responses across all conditions:

```bash
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --cochlea_file data/cochlea_convolved.pkl \
  --analyze_cochlea_correlations
```

**Output**: Correlation analysis between cochlea and mean BEZ responses for each condition.

### 3. Analyze All Conditions
Process all available conditions in the dataset:

```bash
python statistical_testing_bez_runs.py \
  --bez_file  \
  --analyze_all_conditions
```

**Output**: Statistical analysis for every condition in the dataset.

## Advanced Usage Examples

### 4. Save Results and Generate Plots
Save analysis results and generate publication-quality plots:

```bash
python /home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/convolution_HRF/statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --analyze_single_condition \
  --fiber_type hsr \
  --cf_val 1000 \
  --freq_val 1000 \
  --db_val 60 \
  --save_plots \
  --save_results \
  --output_dir results/
```

### 5. Custom Correlation Parameters
Use Spearman correlation with custom outlier threshold:

```bash
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --analyze_single_condition \
  --fiber_type msr \
  --cf_val 2000 \
  --freq_val 2000 \
  --db_val 80 \
  --correlation_type spearman \
  --outlier_threshold 2.5
```

### 6. Analyze Specific Fiber Types Only
Analyze only high and medium spontaneous rate fibers:

```bash
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --cochlea_file data/cochlea_convolved.pkl \
  --analyze_cochlea_correlations \
  --fiber_types hsr msr
```

### 7. Custom Plot Settings
Customize plot appearance and save as PDF:

```bash
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --analyze_single_condition \
  --fiber_type lsr \
  --cf_val 500 \
  --freq_val 500 \
  --db_val 40 \
  --save_plots \
  --plot_format pdf \
  --figsize 14 10 \
  --output_dir my_results/
```

### 8. Verbose Analysis with Progress Tracking
Enable detailed progress information for large datasets:

```bash
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --analyze_all_conditions \
  --verbose \
  --save_results \
  --output_dir batch_analysis/
```

## Complete Analysis Example

Full analysis with cochlea comparison and all options:

```bash
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --cochlea_file data/cochlea_convolved.pkl \
  --analyze_single_condition \
  --fiber_type hsr \
  --cf_val 1000 \
  --freq_val 1000 \
  --db_val 60 \
  --correlation_type pearson \
  --outlier_threshold 2.0 \
  --tr 0.1 \
  --save_plots \
  --save_results \
  --plot_format png \
  --figsize 12 8 \
  --output_dir analysis_results/ \
  --verbose
```

## Parameter Reference

### Required Parameters
- `--bez_file`: Path to BEZ convolved responses pickle file

### Analysis Modes (choose one)
- `--analyze_single_condition`: Analyze one specific condition (generates pairwise correlation matrix)
- `--analyze_cochlea_correlations`: Compare cochlea with BEZ responses across all conditions
- `--analyze_all_conditions`: Process all available conditions in the dataset

### Single Condition Parameters (required with --analyze_single_condition)
- `--fiber_type`: Fiber type to analyze (`hsr`, `msr`, or `lsr`)
- `--cf_val`: Characteristic frequency value in Hz
- `--freq_val`: Stimulus frequency value in Hz  
- `--db_val`: Sound pressure level in dB

### Optional Analysis Parameters
- `--cochlea_file`: Path to cochlea convolved responses pickle file
- `--fiber_types`: Select fiber types to analyze (default: `hsr msr lsr`)
- `--correlation_type`: Correlation method (`pearson` or `spearman`, default: `pearson`)
- `--outlier_threshold`: Z-score threshold for outlier detection (default: `2.0`)
- `--tr`: Temporal resolution in seconds (default: `0.1`)

### Output Options
- `--output_dir`: Output directory for results and plots (default: `statistical_results`)
- `--save_plots`: Save generated plots to files
- `--save_results`: Save analysis results to pickle files
- `--plot_format`: File format for plots (`png`, `pdf`, or `svg`, default: `png`)
- `--figsize`: Figure dimensions in inches (width height, default: `12 8`)
- `--verbose`: Enable detailed progress output

## Understanding the Output

### Pairwise Correlation Matrix
When using `--analyze_single_condition`, you'll get:

1. **Correlation Matrix Plot**: Shows correlations between all pairs of runs
2. **Distribution Plot**: Histogram of pairwise correlation values
3. **Statistical Summary**: Mean, standard deviation, and range of correlations
4. **Cochlea Comparison**: If cochlea data provided, shows how cochlea correlates with mean BEZ

### Key Metrics Explained
- **Mean Pairwise Correlation**: Average correlation between all run pairs
- **Standard Deviation**: Variability in correlations (lower = more consistent)
- **Cochlea-BEZ Correlation**: How well cochlea predicts the mean BEZ response

### Output Files

When `--save_results` is enabled:
- **Single condition**: `single_condition_results_[fiber]_[cf]Hz_[freq]Hz_[db]dB_[timestamp].pkl`
- **Cochlea correlations**: `cochlea_bez_mean_correlations_[timestamp].pkl`
- **All conditions**: `all_conditions_results_[timestamp].pkl`

When `--save_plots` is enabled:
- **Main analysis**: `statistical_analysis_[params]_[timestamp].[format]`
- **Correlation matrix**: `pairwise_correlation_matrix_[params]_[timestamp].png`
- **Distribution**: `pairwise_correlation_distribution_[params]_[timestamp].png`

## Common Use Cases

### 1. Quality Check for Specific Condition
```bash
# Check how consistent runs are for a specific condition
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --analyze_single_condition \
  --fiber_type hsr --cf_val 1000 --freq_val 1000 --db_val 60
```

### 2. Model Validation Against Cochlea
```bash
# See how well BEZ model matches cochlea across all conditions
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --cochlea_file data/cochlea_convolved.pkl \
  --analyze_cochlea_correlations \
  --save_results
```

### 3. Comprehensive Dataset Analysis
```bash
# Analyze everything and save for later review
python statistical_testing_bez_runs.py \
  --bez_file data/bez_convolved.pkl \
  --analyze_all_conditions \
  --save_results --verbose \
  --output_dir comprehensive_analysis/
```

## Tips for Interpretation

- **High pairwise correlations (>0.8)**: Runs are very consistent
- **Low pairwise correlations (<0.5)**: High variability between runs, may indicate noise or instability
- **Cochlea-BEZ correlation**: Measures how well the model captures biological responses
- **Significant p-values (<0.05)**: Statistically reliable correlations

## Troubleshooting

### Common Issues
1. **No data found**: Check that fiber type, CF, frequency, and dB values exist in your data
2. **Insufficient runs**: Need at least 2 runs for pairwise analysis
3. **File not found**: Ensure pickle file paths are correct
4. **Memory issues**: Use `--verbose` to track progress and consider analyzing subsets

### Getting Help
```bash
python statistical_testing_bez_runs.py --help
```

