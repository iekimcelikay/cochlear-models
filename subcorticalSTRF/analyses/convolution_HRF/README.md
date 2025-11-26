# Cochlea & BEZ PSTH-HRF Convolution Analysis

## Updates:
- We want to convolve the WHOLE of HRF with the 200ms PSTH data (23.10.25).
- So generate 20 seconds of HRF data (with 0.1s TR) and convolve with 200ms PSTH data. Don't add zeros after 200ms PSTH data.

## Overview

This script performs convolution analysis between auditory nerve model peristimulus time histograms (PSTH) and the canonical hemodynamic response function (HRF) to predict BOLD fMRI signals from auditory nerve firing patterns. It supports both **Cochlea** and **BEZ2018** auditory nerve models for comprehensive analysis and comparison.

**Pipeline Status: ✅ FULLY FUNCTIONAL** - All model options (cochlea, bez, both) are working and tested.

- **Neural Activity**: Measured in milliseconds (our PSTH data spans 200ms)
- **BOLD Response**: Develops over 15-20 seconds following neural activity

## What This Script Does

### 1. **Multi-Model Data Loading**
- **Cochlea Model**: Loads from `cochlea_meanrate/out/condition_psths/cochlea_psths.mat`
- **BEZ2018 Model**: Loads from `BEZ2018_meanrate/results/processed_data/bez_acrossruns_psths.mat`
- **Both Models**: Supports simultaneous processing for direct comparison
- Extracts firing rate data for three fiber types:
  - **HSR** (High Spontaneous Rate)
  - **MSR** (Medium Spontaneous Rate) 
  - **LSR** (Low Spontaneous Rate)

### 2. **HRF Generation & BEZ Data Integration**
- Uses the Glover model to generate canonical HRF
- **BEZ Data Access**: Correctly extracts data using `bez_data['bez_rates'].lsr/msr/hsr` structure
- **Data Validation**: Automatically checks parameter consistency between models
- Default parameters:
  - **TR**: 0.1 seconds (100ms temporal resolution matching PSTH)
  - **HRF Duration**: 32 seconds (captures full hemodynamic response)
  - **Stimulus Duration**: 200ms (duration of auditory stimuli)

### 3. **Convolution Process**
The key insight is handling the temporal mismatch:

```
200ms Neural Activity → 20-second BOLD Response
```

For each condition (Model × CF × Frequency × dB level × Fiber Type):
1. Take 200ms PSTH data (neural firing rates)
2. Convolve with the canonical HRF

### 4. **Data Organization**
Results are organized hierarchically:
```
# Single model
convolved_responses[fiber_type][cf_frequency][tone_frequency][db_level] = bold_prediction

# Both models
results[model_name][fiber_type][cf_frequency][tone_frequency][db_level] = bold_prediction
```

Where:
- **model_name**: 'cochlea', 'bez' (when processing both models)
- **fiber_type**: 'hsr', 'msr', 'lsr'
- **cf_frequency**: Characteristic frequency of the auditory nerve fiber
- **tone_frequency**: Frequency of the presented tone stimulus
- **db_level**: Sound pressure level of the stimulus

## Pipeline Validation & Testing

### ✅ All Model Options Tested and Working
The complete analysis pipeline has been validated with all three model options:

1. **Cochlea Model Only** (`--model cochlea`) ✅
   - Successfully processes HSR, MSR, LSR fiber types
   - Loads data from `cochlea_meanrate/out/condition_psths/cochlea_psths.mat`
   - Handles 20 CFs × 20 frequencies × 4 dB levels

2. **BEZ Model Only** (`--model bez`) ✅
   - Successfully processes HSR, MSR, LSR fiber types
   - Loads data from `BEZ2018_meanrate/results/processed_data/bez_acrossruns_psths.mat`
   - Uses correct data structure: `bez_data['bez_rates'].lsr/msr/hsr`

3. **Both Models** (`--model both`) ✅
   - Processes both models simultaneously for comparison
   - Handles parameter differences between models gracefully
   - Saves results in organized directory structure

### Recent Fixes Applied
- **BEZ Data Extraction**: Fixed to use correct `bez_rates` structure from regression analysis
- **Function Signatures**: Aligned all HRF generation calls with correct parameter names
- **Model Selection Logic**: Implemented complete branching for all three model options
- **Parameter Validation**: Added consistency checks between BEZ and Cochlea models

## Features

### Flexible Data Processing
- **Subset Testing**: Process only portions of data for quick testing
- **Argument Parsing**: Configure processing via command line
- **Multiple Models**: Support for both Cochlea and BEZ models

### Command Line Options

#### Basic Usage
```bash
# Run with default settings (Cochlea model, all data)
python convolutionHRF_draft_221025.py

# Process BEZ model instead
python convolutionHRF_draft_221025.py --model bez

# Process both models for comparison
python convolutionHRF_draft_221025.py --model both

# Quick test with minimal data (Cochlea model)
python convolutionHRF_draft_221025.py --test

# Quick test with BEZ model
python convolutionHRF_draft_221025.py --test --model bez

# Process subset of data with specific fiber types
python convolutionHRF_draft_221025.py --max-cfs 5 --max-freqs 10 --fiber-types hsr msr
```

#### Advanced Options
```bash
# Custom HRF parameters with BEZ model
python convolutionHRF_draft_221025.py --model bez --tr 0.05 --hrf-length 15.0

# Process both models, skip plotting for large datasets
python convolutionHRF_draft_221025.py --model both --no-plot

# Display plots without saving them (default behavior)
python convolutionHRF_draft_221025.py --model both

# Save plots to files in addition to displaying them
python convolutionHRF_draft_221025.py --model both --save-plots

# Save plots in PDF format for publication
python convolutionHRF_draft_221025.py --model both --save-plots --plot-format pdf

# Save plots in high-quality SVG format
python convolutionHRF_draft_221025.py --model both --save-plots --plot-format svg

# Custom output directory with model comparison and plot saving
python convolutionHRF_draft_221025.py --model both --output-dir /path/to/results --save-plots
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | choice | cochlea | Which model to process ('cochlea', 'bez', 'both') |
| `--max-cfs` | int | all | Maximum number of CFs to process |
| `--max-freqs` | int | all | Maximum number of frequencies to process |
| `--max-dbs` | int | all | Maximum number of dB levels to process |
| `--fiber-types` | list | all | Fiber types to process ('hsr', 'msr', 'lsr') |
| `--tr` | float | 0.1 | Temporal resolution (seconds) |
| `--hrf-length` | float | 20.0 | HRF duration (seconds) |
| `--db-level` | float | 60.0 | dB level for plotting |
| `--output-dir` | str | ./output | Output directory for results |
| `--no-plot` | flag | False | Skip plotting |
| `--no-save` | flag | False | Skip saving results |
| `--save-plots` | flag | False | Save plots to files (in addition to displaying them) |
| `--plot-format` | choice | png | Format for saved plots ('png', 'pdf', 'svg') |
| `--test` | flag | False | Run in test mode (2 CFs, 3 freqs, 2 dBs) |

## Output

### Saved Files
**Data Files (with timestamps):**
- `convolved_responses_hsr_YYYYMMDD_HHMMSS.pkl`: HSR fiber predictions
- `convolved_responses_msr_YYYYMMDD_HHMMSS.pkl`: MSR fiber predictions  
- `convolved_responses_lsr_YYYYMMDD_HHMMSS.pkl`: LSR fiber predictions
- `combined_models_results_YYYYMMDD_HHMMSS.pkl`: Both models (when using `--model both`)

**Plot Files (when using `--save-plots`):**
- `canonical_hrf_YYYYMMDD_HHMMSS.{png/pdf/svg}`: Hemodynamic response function plot
- `{MODEL}_hrf_responses_{FIBER}_YYYYMMDD_HHMMSS.{png/pdf/svg}`: Convolved responses for each model and fiber type
  - Examples: `COCHLEA_hrf_responses_hsr_20251023_143052.png`
  - Examples: `BEZ_hrf_responses_msr_20251023_143052.pdf`

**Both Models Processing:**
- Files contain nested structure: `results['cochlea'/'bez'][fiber_type][cf][freq][db]`
- Enables direct comparison between Cochlea and BEZ2018 model predictions
- Separate plot files for each model when using `--save-plots`

### Visualizations
1. **Canonical HRF Plot**: Shows the hemodynamic response function used for convolution
2. **Convolved Responses**: BOLD predictions for each fiber type and model
   - **All CFs displayed**: No artificial limits on characteristic frequencies shown
   - **All tone frequencies**: Displays responses to complete frequency range
   - **Dynamic layout**: Adjusts subplot arrangements based on data size
   - **Enhanced legends**: Multiple columns with optimized spacing for large datasets
   - Organized by model, fiber type, and sound level
   - Supports both single-model and comparative visualizations

## File Structure

```
subcorticalSTRF/
├── convolution_HRF/
│   ├── convolutionHRF_draft_221025.py  # Main script
│   ├── README.md                       # This documentation
│   └── output/                         # Results directory
├── cochlea_meanrate/
│   └── out/condition_psths/
│       └── cochlea_psths.mat          # Input data
└── BEZ2018_meanrate/
    └── results/processed_data/
        └── bez_acrossruns_psths.mat   # Alternative input data
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `load_cochlea_psth_data()` | Load Cochlea model PSTH data |
| `load_bez_psth_data()` | Load BEZ2018 model PSTH data |
| `load_both_models_data()` | Load both models simultaneously |
| `generate_canonical_hrf()` | Create HRF using Glover model |
| `convolve_brief_stimulus_with_hrf()` | Handle 200ms→20s convolution |
| `convolve_cochlea_with_hrf()` | Process Cochlea model conditions |
| `convolve_bez_with_hrf()` | Process BEZ model conditions |
| `convolve_both_models_with_hrf()` | Process both models for comparison |
| `plot_convolved_responses()` | Visualize all CFs and frequencies with dynamic layout |
| `save_convolved_data()` | Save predictions to files |
| `parse_arguments()` | Handle command line arguments |
| `subset_parameters()` | Create data subsets for testing |

## Scientific Applications

### 1. **fMRI Prediction**
Predict what BOLD signals would look like in auditory cortex based on peripheral auditory nerve activity.

### 2. **Model Comparison**
Compare different auditory nerve models (Cochlea vs BEZ) in terms of their predicted fMRI responses.

### 3. **Frequency Tuning Analysis**
Understand how different characteristic frequencies and tone presentations translate to hemodynamic responses.

### 4. **Fiber Type Analysis**
Examine how different spontaneous rate fiber types contribute to the overall BOLD signal.

## Dependencies

- `numpy`: Numerical computations
- `scipy`: MATLAB file loading
- `matplotlib`: Plotting
- `nipy`: HRF generation (Glover model)
- `pathlib`: File handling
- `argparse`: Command line parsing

## Example Workflow

### Quick Start
1. **Prepare Data**: Ensure PSTH data files are in correct directories
2. **Test Run**: `python convolutionHRF_draft_221025.py --test`
3. **Model-Specific Analysis**: 
   - Cochlea: `python convolutionHRF_draft_221025.py --model cochlea`
   - BEZ: `python convolutionHRF_draft_221025.py --model bez`
   - Both: `python convolutionHRF_draft_221025.py --model both`
4. **Analyze Results**: Load saved `.pkl` files for further analysis

### Complete Analysis Pipeline
```bash
# Step 1: Quick test to verify setup
python convolutionHRF_draft_221025.py --test --no-save

# Step 2: Process both models with subset for development
python convolutionHRF_draft_221025.py --model both --max-cfs 5 --max-freqs 10

# Step 3: Full analysis with both models
python convolutionHRF_draft_221025.py --model both

# Step 4: Generate specific outputs for publication
python convolutionHRF_draft_221025.py --model both --no-plot --output-dir ./publication_results
```

## Notes

- **Temporal Resolution**: The script uses 100ms bins matching the original PSTH data
- **Convolution Method**: Uses `numpy.convolve` with 'same' mode to preserve timing
- **Memory Efficiency**: Subset processing helps manage large datasets
- **Extensibility**: Easy to add new auditory nerve models or HRF variants

This analysis bridges computational auditory neuroscience with neuroimaging, enabling predictions about how peripheral auditory processing translates to measurable brain signals in fMRI experiments.
