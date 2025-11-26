# Quick Reference: Refactored Loggers Usage

**Created:** 2025-11-26 16:16:00  
**Last Updated:** 2025-11-26 16:16:00

Quick reference guide for using the refactored loggers module across different use cases.

## For WSR Model Scripts

### Running WSR Model with New Loggers
```python
from models.WSRmodel.run_wsr_model_OOP import WSRConfig, WSRSimulation

# Configure (API unchanged)
config = WSRConfig(
    db_range=60,
    freq_range=(125, 2500, 20),
    tone_duration=0.2,
)

# Run simulation
simulation = WSRSimulation(config)
results = simulation.run(save_full_response=True)

# Outputs are automatically saved with enhanced metadata and logging
```

### What You Get Automatically
```
results/wsr_model_psth_batch_92chans_8ms_8tc_-1factor_0shift_TIMESTAMP/
├── params.json              # Full parameters
├── stimulus_params.txt      # Parameters in text format
├── wsr_simulation.log       # Complete log (NEW!)
├── simulation_summary.json  # Key statistics (NEW!)
├── wsr_responses.npz        # Response data
├── wsr_results.pkl          # Pickle format
└── channel_cfs.npy          # Channel CFs
```

## For Model Comparison Scripts

### Using Loggers in Comparisons
```python
from loggers import FolderManager, LoggingConfigurator, MetadataSaver

# 1. Create organized output folder
manager = FolderManager(
    base_dir='./results',
    name_builder='model_comparison'
).with_params(
    model1='wsr',
    model2='bez',
    comparison_type='population'
)
output_dir = manager.create_folder()

# 2. Setup logging
LoggingConfigurator.setup_basic(output_dir, 'analysis.log')

# 3. Run your analysis
# ... your code here ...

# 4. Save summary
MetadataSaver().save_json(output_dir, your_results_dict, 'summary.json')
```

## For Any Custom Script

### Option 1: Use Registered Builder
```python
from loggers import FolderManager, LoggingConfigurator

# Use a registered name builder (bez2018, wsr_model, cochlea_zilany2014, model_comparison)
manager = FolderManager('./results', 'wsr_model')
output_dir = manager.with_params(num_channels=128, ...).create_folder()
LoggingConfigurator.setup_basic(output_dir, 'log.txt')
```

### Option 2: Custom Name Builder
```python
from loggers import FolderManager, LoggingConfigurator

def my_namer(params, timestamp):
    return f"{params['analysis']}_{params['version']}_{timestamp}"

manager = FolderManager('./results', my_namer)
output_dir = manager.with_params(analysis='test', version='v1').create_folder()
LoggingConfigurator.setup_basic(output_dir)
```

### Option 3: Simple/Explicit Name
```python
from loggers import FolderManager, LoggingConfigurator

# No name builder - just timestamp
manager = FolderManager('./results')
output_dir = manager.create_folder()  # results_TIMESTAMP

# Or explicit name
output_dir = manager.create_folder('my_analysis_2025')
LoggingConfigurator.setup_basic(output_dir)
```

## Logging Styles

### Basic (Default)
```python
LoggingConfigurator.setup_basic(output_dir, 'log.txt')
# Format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

### Console Only (No File)
```python
LoggingConfigurator.setup_console_only(level=logging.INFO)
# No file output, just console
```

### Detailed (With Line Numbers)
```python
LoggingConfigurator.setup_detailed(output_dir, 'detailed.log')
# Includes filename and line numbers
```

### Custom Configuration
```python
config = LoggingConfigurator(
    output_dir=output_dir,
    log_filename='custom.log',
    file_level=logging.DEBUG,
    console_level=logging.WARNING,
    format_string='%(levelname)s: %(message)s'
)
config.setup()
```

## Saving Metadata

### JSON Format (Recommended)
```python
from loggers import MetadataSaver

saver = MetadataSaver()
data = {
    'r_squared': 0.95,
    'p_value': 1.23e-10,
    'parameters': ['param1', 'param2']
}
saver.save_json(output_dir, data, 'results.json')
```

### Text Format
```python
saver.save_text(output_dir, data, 'results.txt')
# Creates: param1=value1\nparam2=value2...
```

### YAML Format (Requires PyYAML)
```python
saver.save_yaml(output_dir, data, 'results.yaml')
```

## Common Patterns

### Full Analysis Workflow
```python
from loggers import FolderManager, LoggingConfigurator, MetadataSaver
import logging

# Setup
manager = FolderManager('./results', 'model_comparison')
output_dir = manager.with_params(model1='A', model2='B').create_folder()
LoggingConfigurator.setup_basic(output_dir, 'analysis.log')
logger = logging.getLogger(__name__)

# Run analysis
logger.info("Starting analysis...")
results = run_your_analysis()
logger.info(f"Analysis complete: R²={results['r2']:.3f}")

# Save results
MetadataSaver().save_json(output_dir, results, 'summary.json')
logger.info(f"Results saved to {output_dir}")
```

### Subfolders for Organization
```python
manager = FolderManager('./results', 'my_analysis')
output_dir = manager.with_params(version=1).create_folder()

# Create subfolders
figures_dir = manager.create_subfolder('figures')
data_dir = manager.create_subfolder('data')

# Save to subfolders
plt.savefig(f"{figures_dir}/plot.png")
np.save(f"{data_dir}/results.npy", data)
```

## Migration Checklist

- [ ] Replace `ExperimentFolderManager` → `FolderManager`
- [ ] Replace `create_batch_folder()` → `create_folder()`
- [ ] Add `LoggingConfigurator.setup_basic()` at start
- [ ] Replace `print()` with `logger.info()`, `logger.warning()`, etc.
- [ ] Use `MetadataSaver` for JSON/text output
- [ ] Add more params to `with_params()` for better tracking

## Tips

1. **Always setup logging early** - Right after creating output folder
2. **Use logger instead of print** - Gets automatically saved to file
3. **Save metadata as JSON** - Easy to parse and analyze later
4. **Track all parameters** - Use `with_params()` extensively
5. **Use subfolder for organization** - Separate figures, data, logs, etc.
