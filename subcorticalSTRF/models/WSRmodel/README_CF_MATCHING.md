# CF Matching for WSR Model

## Overview

The WSR (Wang & Shamma) auditory model uses a fixed filterbank with 128 channels spanning the frequency range. However, when comparing WSR outputs with cochlea and BEZ2018 models, we need to extract only the channels that match the specific characteristic frequencies (CFs) used in those models.

This module provides the `CFMatcher` class to handle the mapping between WSR channels and target CFs.

## Problem

- **WSR Model**: Fixed 128 channels with specific center frequencies
- **Cochlea/BEZ Models**: Use specific CFs (e.g., 20 logarithmically-spaced CFs from 125 Hz to 4 kHz)
- **Challenge**: WSR channel frequencies don't exactly match cochlea/BEZ CFs

## Solution

The `CFMatcher` class:
1. Maps each target CF to the nearest WSR channel (using log-frequency distance)
2. Extracts only those WSR channels for comparison
3. Validates that mappings are within acceptable tolerance

## Quick Start

### Basic Usage

```python
from subcorticalSTRF.WSRmodel.wsrmodel import WSRModel
from subcorticalSTRF.WSRmodel.cf_matcher import create_cf_matcher_from_models
import numpy as np

# Create WSR model
wsr_params = [8, 8, -1, 0]
wsr_model = WSRModel(wsr_params)

# Define target CFs (e.g., from cochlea data)
params = {'cfs': np.array([125, 250, 500, 1000, 2000, 4000])}

# Create CF matcher
fs = 16000
matcher = create_cf_matcher_from_models(wsr_model, params, fs)

# View mapping
print(matcher.get_mapping_summary())
```

### Extract Channels at Target CFs

```python
# Generate audio
from subcorticalSTRF.stim_generator.soundgen import SoundGen
soundgen = SoundGen(16000, tau=0.005)
tone = soundgen.sound_maker(freq=500, tone_duration=0.2)

# Process with WSR (all 128 channels)
wsr_response = wsr_model.process(tone, fs)

# Extract only channels matching target CFs
extracted_channels, corresponding_cfs = matcher.extract_channels(wsr_response)

# Compute temporal mean (for comparison with cochlea/BEZ)
temporal_mean = np.mean(extracted_channels, axis=0)
```

### Using Convenience Method

The `WSRModel` class has a built-in convenience method:

```python
# Process and extract in one step
extracted_channels, cfs = wsr_model.process_at_target_cfs(tone, fs, target_cfs)
temporal_mean = np.mean(extracted_channels, axis=0)
```

## Complete Workflow Example

### Matching Cochlea/BEZ Data Structure

```python
from scipy.io import loadmat
import numpy as np
from subcorticalSTRF.WSRmodel.wsrmodel import WSRModel
from subcorticalSTRF.stim_generator.soundgen import SoundGen

# 1. Load target CFs from cochlea/BEZ data
data = loadmat('cochlea_psths.mat')
target_cfs = data['cfs']
frequencies = data['frequencies']
dbs = data['dbs']

# 2. Create WSR model
wsr_model = WSRModel([8, 8, -1, 0])
soundgen = SoundGen(16000, tau=0.005)
fs = 16000

# 3. Create output array (matching cochlea/BEZ structure)
wsr_rates = np.zeros((len(target_cfs), len(frequencies), len(dbs)))

# 4. Process each condition
for freq_idx, freq in enumerate(frequencies):
    for db_idx, db in enumerate(dbs):
        # Generate tone (scale for dB level as needed)
        tone = soundgen.sound_maker(freq=freq, tone_duration=0.2)
        
        # Process at target CFs
        channels, _ = wsr_model.process_at_target_cfs(tone, fs, target_cfs)
        
        # Compute temporal mean
        temporal_mean = np.mean(channels, axis=0)
        
        # Store in cochlea/BEZ structure
        wsr_rates[:, freq_idx, db_idx] = temporal_mean

# Now wsr_rates has shape (n_CFs, n_frequencies, n_dBs)
# This matches the cochlea/BEZ data structure!
```

## API Reference

### CFMatcher Class

```python
class CFMatcher:
    """Maps WSR channel frequencies to target CFs."""
    
    def __init__(self, wsr_cfs: np.ndarray, target_cfs: np.ndarray):
        """
        Args:
            wsr_cfs: WSR filterbank center frequencies (Hz)
            target_cfs: Target CFs from cochlea/BEZ (Hz)
        """
    
    def get_wsr_channel(self, target_cf: float) -> int:
        """Get WSR channel index for a given target CF."""
    
    def get_target_cf(self, wsr_channel: int) -> float:
        """Get nearest target CF for a WSR channel index."""
    
    def extract_channels(
        self, 
        wsr_response: np.ndarray, 
        target_cfs: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract WSR channels corresponding to target CFs.
        
        Args:
            wsr_response: WSR response (n_time, n_channels)
            target_cfs: Optional specific CFs to extract
            
        Returns:
            (extracted_channels, corresponding_cfs)
        """
    
    def get_mapping_summary(self) -> str:
        """Get formatted summary of CF mapping."""
    
    def validate_mapping(self, tolerance_percent: float = 15.0) -> bool:
        """Validate all mappings are within tolerance."""
```

### WSRModel Methods

```python
class WSRModel:
    def process_at_target_cfs(
        self, 
        audio: np.ndarray, 
        fs: int, 
        target_cfs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process audio and extract channels matching target CFs.
        
        Returns:
            (extracted_channels, corresponding_cfs)
        """
```

## Mapping Validation

The matcher automatically validates mappings:

```python
# Check if all mappings are within 15% tolerance
is_valid = matcher.validate_mapping(tolerance_percent=15.0)

if not is_valid:
    print("Warning: Some CFs have poor matches!")
```

Mappings with >10% error are logged as warnings.

## Visualization

See the mapping quality:

```python
# Run the example script
python example_cf_matching.py
```

This creates a visualization showing:
- WSR channel CFs with target CF mappings overlaid
- Matching accuracy (percent error) for each target CF

## Files

- `cf_matcher.py` - CFMatcher class implementation
- `wsrmodel.py` - WSRModel with `process_at_target_cfs()` method
- `example_cf_matching.py` - Comprehensive usage examples
- `test_cf_matcher.py` - Quick test suite

## Testing

Run the test suite:

```bash
cd subcorticalSTRF/WSRmodel
python test_cf_matcher.py
```

Run the full examples:

```bash
python example_cf_matching.py
```

## Notes

- **Logarithmic matching**: The matcher uses log-frequency distance to find nearest WSR channels, which is more perceptually appropriate for auditory frequencies.

- **Tolerance**: By default, mappings >15% different trigger warnings. Most mappings should be <5% different.

- **Data structure compatibility**: The output of `process_at_target_cfs()` followed by `np.mean(axis=0)` produces arrays with the same structure as cochlea/BEZ temporal means, enabling direct comparison.

- **Performance**: CF matching happens once per matcher creation. Extracting channels is a simple array indexing operation.

## Common Use Cases

### 1. Compare WSR with Cochlea/BEZ at specific CFs
```python
wsr_mean = wsr_model.process_at_target_cfs(tone, fs, target_cfs)[0].mean(axis=0)
# wsr_mean now comparable with cochlea_mean_rates
```

### 2. Regression analysis across models
```python
# Extract WSR rates at cochlea CFs
wsr_rates = []
cochlea_rates = []

for cf in target_cfs:
    channels, _ = wsr_model.process_at_target_cfs(tone, fs, [cf])
    wsr_rates.append(channels.mean())
    cochlea_rates.append(cochlea_data[cf])

# Now perform regression
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(cochlea_rates, wsr_rates)
```

### 3. Generate WSR tuning curves
```python
# Process multiple frequencies at multiple CFs
for cf in target_cfs:
    rates = []
    for freq in frequencies:
        tone = soundgen.sound_maker(freq=freq, tone_duration=0.2)
        channels, _ = wsr_model.process_at_target_cfs(tone, fs, [cf])
        rates.append(channels.mean())
    plot_tuning_curve(frequencies, rates, cf)
```
