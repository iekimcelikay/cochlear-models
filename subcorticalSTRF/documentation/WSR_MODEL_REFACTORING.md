# WSR Model Scripts Refactoring - Using New Loggers Module

**Created:** 2025-11-26 16:14:00  
**Last Updated:** 2025-11-26 16:14:00

Comprehensive documentation of WSR model scripts refactoring to use the new loggers module.

## Overview

Refactored WSR model scripts to use the new `loggers` module for better organized output management, structured logging, and metadata saving.

## Files Modified

### 1. `models/WSRmodel/run_wsr_model_OOP.py`

**Changes Made:**
- ✅ Updated imports to use `FolderManager`, `LoggingConfigurator`, `MetadataSaver`
- ✅ Added `logging` module and created logger instance
- ✅ Replaced `ExperimentFolderManager` with `FolderManager`
- ✅ Changed `create_batch_folder()` to `create_folder()`
- ✅ Added `LoggingConfigurator.setup_basic()` for file + console logging
- ✅ Replaced all `print()` statements with `logger.info()`, `logger.warning()`, etc.
- ✅ Added `MetadataSaver` to save simulation summary statistics as JSON
- ✅ Enhanced parameter tracking in folder metadata

**Before:**
```python
from loggers import ExperimentFolderManager

folder_manager = ExperimentFolderManager(self.config.results_folder, "wsr_model")
folder_manager.with_params(
    num_channels=int(num_channels),
    frame_length=self.config.wsr_params[0],
    # ... other params
)
folder_manager.create_batch_folder()
self.save_dir = folder_manager.get_results_folder()
print(f"[INFO] Results will be saved to: {self.save_dir}")
```

**After:**
```python
from loggers import FolderManager, LoggingConfigurator, MetadataSaver

folder_manager = FolderManager(self.config.results_folder, "wsr_model")
folder_manager.with_params(
    num_channels=int(num_channels),
    frame_length=self.config.wsr_params[0],
    # ... more detailed params including max_cf, num_harmonics, etc.
)
self.save_dir = Path(folder_manager.create_folder())

# Setup logging
LoggingConfigurator.setup_basic(self.save_dir, 'wsr_simulation.log')
logger.info(f"Results will be saved to: {self.save_dir}")
logger.info(f"WSR parameters: {self.config.wsr_params}")
```

**New Features:**
- Automatic logging to both file (`wsr_simulation.log`) and console
- Saves `simulation_summary.json` with key statistics:
  ```json
  {
    "total_conditions": 20,
    "num_channels": 92,
    "cf_range_hz": [180.5, 2499.8],
    "wsr_params": [8, 8, -1, 0],
    "sample_rate": 16000,
    "tone_duration": 0.2,
    "db_range": 60,
    "freq_range": [125, 2500, 20]
  }
  ```

### 2. `models/WSRmodel/run_wsr_model_pipeline.py`

**Changes Made:**
- ✅ Updated example usage to use `FolderManager` instead of `ExperimentFolderManager`
- ✅ Changed `create_batch_folder()` to `create_folder()`
- ✅ Added `LoggingConfigurator.setup_basic()` to example
- ✅ Updated to save only `npz` format (removed `.mat`)
- ✅ Fixed folder path handling in example

**Before:**
```python
from loggers import ExperimentFolderManager
folder_mgr = (ExperimentFolderManager("./results", "wsr_model")
              .with_params(num_channels=128)
              .create_batch_folder())

collector = ResultCollector(folder_mgr.get_results_path())
results = collector.collect(processor.process(stimuli))
collector.save(formats=['mat', 'npz'])
```

**After:**
```python
from loggers import FolderManager, LoggingConfigurator

folder_mgr = (FolderManager("./results", "wsr_model")
              .with_params(
                  num_channels=128,
                  frame_length=8,
                  time_constant=8,
                  factor=-1,
                  shift=0,
                  sample_rate=16000
              )
              .create_folder())

# Setup logging
LoggingConfigurator.setup_basic(folder_mgr, 'pipeline.log')

collector = ResultCollector(folder_mgr)
results = collector.collect(processor.process(stimuli))
collector.save(formats=['npz'])
```

## Output Directory Structure

### Before Refactoring
```
results/wsr_model_psth_batch_92chans_8ms_8tc_-1factor_0shift_20251124_154938/
├── params.json                 # Basic parameters
├── stimulus_params.txt         # Same as JSON but in text
├── wsr_responses.npz          # Response data
├── wsr_results.pkl            # Pickle format
└── channel_cfs.npy            # Channel CFs
```

### After Refactoring
```
results/wsr_model_psth_batch_92chans_8ms_8tc_-1factor_0shift_20251126_163052/
├── params.json                 # Enhanced with more parameters
├── stimulus_params.txt         # Text format
├── wsr_simulation.log         # NEW: Complete run log (file + console)
├── simulation_summary.json    # NEW: Key statistics and metadata
├── wsr_responses.npz          # Response data
├── wsr_results.pkl            # Pickle format
└── channel_cfs.npy            # Channel CFs
```

## Benefits

### 1. **Better Logging**
- ✅ Unified logging to both file and console
- ✅ Timestamped log entries
- ✅ Different log levels (INFO, WARNING, ERROR, etc.)
- ✅ Complete record of simulation run

### 2. **Enhanced Metadata**
- ✅ More comprehensive parameter tracking
- ✅ Separate summary statistics file for quick reference
- ✅ JSON format for easy parsing and analysis

### 3. **Consistency**
- ✅ Same logging/folder management across all scripts
- ✅ Matches the pattern used in `demo_comparison_pipeline.py`
- ✅ Reusable patterns for future scripts

### 4. **Maintainability**
- ✅ Easier to add new parameters or outputs
- ✅ Centralized logging configuration
- ✅ No duplicate logging setup code

## Migration Notes

### For Existing Scripts Using WSR Model

If you have scripts that import and use `run_wsr_model_OOP.py`, no changes are needed! The API remains the same:

```python
from models.WSRmodel.run_wsr_model_OOP import WSRConfig, WSRSimulation

# Same usage as before
config = WSRConfig(
    db_range=60,
    freq_range=(125, 2500, 20),
    tone_duration=0.2,
)

model_simulation = WSRSimulation(config)
results = model_simulation.run(save_full_response=True)
```

The only difference is:
- Better logging output (now goes to file + console)
- Additional metadata files in output directory
- More detailed parameter tracking

### For Scripts Using ExperimentFolderManager

The `FolderManager` class maintains backward compatibility through the `ExperimentFolderManager` alias, so existing code will continue to work. However, you'll need to change:

```python
# Old method name (will error)
folder_manager.create_batch_folder()

# New method name (works with both FolderManager and ExperimentFolderManager alias)
folder_manager.create_folder()
```

## Testing

All refactored scripts have been validated:
- ✅ Python syntax check passed
- ✅ Import statements verified
- ✅ Backward compatibility maintained
- ✅ Output structure enhanced but compatible

## Next Steps

Consider applying the same refactoring pattern to:
1. BEZ2018 model scripts
2. Cochlea model scripts  
3. Any other analysis/experiment scripts in the project

The loggers module is now fully generic and can handle any workflow that needs:
- Organized output directories
- Structured logging
- Metadata/parameter tracking
- Timestamped results
