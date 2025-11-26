# Loggers Module Refactoring Summary

## Changes Made

### 1. **FolderManager Refactoring** (`folder_manager.py`)
   - **Renamed**: `ExperimentFolderManager` → `FolderManager` (with backward compatibility alias)
   - **Made more flexible**: 
     - `name_builder` parameter is now optional
     - Can accept string (from registry), callable function, or None (default timestamp-based naming)
     - `create_batch_folder()` → `create_folder()` with optional explicit folder name
   - **Use cases**: Now supports experiments, model comparisons, analyses, or any workflow needing organized outputs

### 2. **New LoggingConfigurator** (`logging_configurator.py`)
   - **Purpose**: Reusable logging setup for file + console output
   - **Features**:
     - Multiple format presets (DEFAULT, SIMPLE, DETAILED)
     - Configurable log levels for file and console separately
     - Class methods for quick setup: `setup_basic()`, `setup_console_only()`, `setup_detailed()`
   - **Replaces**: Custom logging setup code in individual scripts

### 3. **Model Comparison Builder** (`model_builders.py`)
   - **Added**: `model_comparison` name builder for comparison analyses
   - **Parameters**: `model1`, `model2`, `comparison_type`
   - **Example output**: `wsr_vs_bez_population_20251126_143052`

### 4. **Updated demo_comparison_pipeline.py**
   - **Now uses**:
     - `FolderManager` for output directory creation with `model_comparison` builder
     - `LoggingConfigurator` for logging setup
     - `MetadataSaver` for saving summary statistics
   - **Benefits**:
     - Automatic timestamped folder creation
     - Consistent logging across runs
     - Metadata saved as JSON for easy analysis
     - Cleaner, more maintainable code

### 5. **Updated Package Exports** (`__init__.py`)
   - Exported `FolderManager`, `LoggingConfigurator` as main interfaces
   - Added comprehensive documentation with usage examples
   - Maintained backward compatibility with `ExperimentFolderManager` alias

## Usage Examples

### Basic Experiment
```python
from loggers import FolderManager, LoggingConfigurator

# Create folder with model-specific naming
manager = (FolderManager("./results", "bez2018")
          .with_params(num_runs=10, num_cf=20, num_ANF=(4,4,4))
          .create_folder())

# Setup logging
LoggingConfigurator.setup_basic(manager.get_results_folder(), 'experiment.log')
```

### Model Comparison
```python
from loggers import FolderManager, LoggingConfigurator, MetadataSaver

# Create comparison output folder
manager = FolderManager("./results", "model_comparison")
output_dir = manager.with_params(
    model1='wsr', 
    model2='bez',
    comparison_type='population'
).create_folder()

# Setup logging and save results
LoggingConfigurator.setup_basic(output_dir, 'comparison.log')
MetadataSaver().save_json(output_dir, results_dict)
```

### Custom Analysis
```python
from loggers import FolderManager, LoggingConfigurator

# Custom naming function
def custom_namer(params, timestamp):
    return f"{params['analysis_type']}_{params['dataset']}_{timestamp}"

manager = FolderManager("./results", custom_namer)
output_dir = manager.with_params(
    analysis_type='regression',
    dataset='neural_data'
).create_folder()

LoggingConfigurator.setup_detailed(output_dir)
```

### Simple Usage (No Custom Naming)
```python
from loggers import FolderManager, LoggingConfigurator

# Just creates a timestamped folder
manager = FolderManager("./results")
output_dir = manager.create_folder()  # Creates: results_20251126_143052

# Or with explicit name
output_dir = manager.create_folder("my_analysis_v1")
```

## Benefits

1. **Reusability**: Logging and folder management code can be reused across all scripts
2. **Consistency**: Standardized output structure and logging format
3. **Flexibility**: Works for experiments, comparisons, analyses, etc.
4. **Maintainability**: Changes to logging/folder structure made in one place
5. **Backward Compatibility**: Existing code using `ExperimentFolderManager` still works

## Migration Guide

### Old Code
```python
# Custom logging setup in each script
def setup_logging(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # ... handler setup ...
    
output_dir = base_dir / 'results/analysis_20251124'
log_file = setup_logging(output_dir)
```

### New Code
```python
from loggers import FolderManager, LoggingConfigurator

# Use loggers module
manager = FolderManager(base_dir / 'results', 'model_comparison')
output_dir = manager.with_params(
    model1='wsr', model2='bez', comparison_type='population'
).create_folder()

log_file = LoggingConfigurator.setup_basic(output_dir, 'analysis.log')
```
