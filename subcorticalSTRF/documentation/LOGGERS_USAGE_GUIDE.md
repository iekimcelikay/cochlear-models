# Using Loggers in Model Comparison Pipeline

This document explains how the refactored `loggers` module is used in `demo_comparison_pipeline.py`.

## Overview

The comparison pipeline now uses three main components from the `loggers` module:

1. **FolderManager** - Creates organized output directories with automatic timestamping
2. **LoggingConfigurator** - Sets up file and console logging
3. **MetadataSaver** - Saves analysis parameters and results to JSON/text

## Code Walkthrough

### 1. Import the Loggers Module

```python
from loggers import FolderManager, LoggingConfigurator, MetadataSaver
```

### 2. Create Output Directory Structure

```python
# Use FolderManager with 'model_comparison' builder
folder_manager = FolderManager(
    base_dir=base_dir / 'model_comparisons/results',
    name_builder='model_comparison'
).with_params(
    model1='wsr',
    model2='bez',
    comparison_type='population',
    wsr_results_path=str(wsr_results_path),
    population_weights={'hsr': 0.65, 'msr': 0.23, 'lsr': 0.12},
    min_cf=180.0,
    fs=16000
)

output_dir = Path(folder_manager.create_folder())
```

**Result**: Creates a timestamped folder like:
```
model_comparisons/results/wsr_vs_bez_population_20251126_143052/
├── params.json              # Auto-saved by FolderManager
└── stimulus_params.txt      # Auto-saved by FolderManager
```

### 3. Setup Logging

```python
# Configure logging to both file and console
log_file = LoggingConfigurator.setup_basic(output_dir, 'comparison_log.txt')
logger.info(f"Log file: {log_file}")
logger.info(f"Output directory: {output_dir}")
```

**Result**: Creates `comparison_log.txt` in the output directory and starts logging to both file and console.

### 4. Run the Comparison Pipeline

```python
# ComparisonPipeline uses the output_dir we created
pipeline = ComparisonPipeline(
    wsr_results_path=wsr_results_path,
    output_dir=output_dir,
    min_cf=180.0,
    fs=16000,
    compute_population=True
)

results = pipeline.run(
    fiber_types=['hsr', 'msr', 'lsr'],
    bez_freq_range=(125, 2500, 20),
    wsr_params=[8, 8, -1, 0]
)
```

### 5. Save Results Using Pipeline Methods

```python
# Pipeline's built-in save methods
pipeline.save_results(results, prefix="wsr_vs_bez_population")
pipeline.plot_results(results, prefix="wsr_vs_bez_population", show_plots=True)
```

**Result**: Pipeline saves its own output files:
```
wsr_vs_bez_population_20251126_143052/
├── wsr_vs_bez_population_cf_mapping.csv
├── wsr_vs_bez_population_regression.npz
├── wsr_vs_bez_population_aligned_data.npz
├── wsr_vs_bez_population_complete_results.pkl
└── wsr_vs_bez_population_comparison.png
```

### 6. Save Summary Statistics Using MetadataSaver

```python
# Extract key results and save as JSON
metadata_saver = MetadataSaver()
summary_stats = {
    'r_squared': float(results.regression_results.r_squared),
    'p_value': float(results.regression_results.p_value),
    'slope': float(results.regression_results.slope),
    'intercept': float(results.regression_results.intercept),
    'n_data_points': len(results.regression_results.x_data),
    'model_comparison': f"{results.metadata['model1']} vs {results.metadata['model2']}",
    'fiber_types': results.metadata['fiber_types'],
    'population_weights': results.metadata['population_weights']
}
metadata_saver.save_json(output_dir, summary_stats, filename="summary_statistics.json")
```

**Result**: Creates `summary_statistics.json` with key results in easily accessible format.

## Final Directory Structure

After running the complete pipeline:

```
model_comparisons/results/wsr_vs_bez_population_20251126_143052/
├── params.json                                    # Parameters used (FolderManager)
├── stimulus_params.txt                            # Parameters in text format (FolderManager)
├── comparison_log.txt                             # Complete run log (LoggingConfigurator)
├── summary_statistics.json                        # Key results summary (MetadataSaver)
├── wsr_vs_bez_population_cf_mapping.csv          # CF alignment data (Pipeline)
├── wsr_vs_bez_population_regression.npz          # Regression results (Pipeline)
├── wsr_vs_bez_population_aligned_data.npz        # Aligned model data (Pipeline)
├── wsr_vs_bez_population_complete_results.pkl    # Full results object (Pipeline)
└── wsr_vs_bez_population_comparison.png          # Comparison plot (Pipeline)
```

## Benefits

### Before Refactoring
```python
# Custom logging setup in script
def setup_logging(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'comparison_log.txt'
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    # ... more boilerplate ...

# Manual directory creation
output_dir = base_dir / 'model_comparisons/results/wsr_vs_bez_population_20251124'
log_file = setup_logging(output_dir)
```

### After Refactoring
```python
# Clean, reusable code
folder_manager = FolderManager(
    base_dir / 'model_comparisons/results',
    'model_comparison'
).with_params(model1='wsr', model2='bez', comparison_type='population')

output_dir = Path(folder_manager.create_folder())
log_file = LoggingConfigurator.setup_basic(output_dir, 'comparison_log.txt')
```

## Key Advantages

1. **Automatic Timestamping**: No more manual date strings in folder names
2. **Reusable Code**: Same logging setup can be used in any script
3. **Organized Outputs**: Consistent structure across all analyses
4. **Parameter Tracking**: Automatically saves params.json for reproducibility
5. **Easy Metadata**: Simple interface for saving summary statistics
6. **Maintainable**: Changes to logging/folder structure made in one place

## Other Use Cases

The same loggers module can be used for:

- BEZ model experiments: `FolderManager("./results", "bez2018")`
- WSR model experiments: `FolderManager("./results", "wsr_model")`
- Cochlea simulations: `FolderManager("./results", "cochlea_zilany2014")`
- Custom analyses: `FolderManager("./results", custom_namer_function)`
- Quick analyses: `FolderManager("./results").create_folder("analysis_v1")`
