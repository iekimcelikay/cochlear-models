"""
Example: Using CFMatcher to Extract WSR Channels for Cochlea/BEZ CFs

This script demonstrates how to:
1. Load cochlea/BEZ parameters to get target CFs
2. Create a WSR model
3. Use CFMatcher to map between WSR channels and target CFs
4. Extract WSR responses at specific CFs for comparison with other models
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

from WSRmodel.wsrmodel import WSRModel
from WSRmodel.cf_matcher import CFMatcher, create_cf_matcher_from_models
from stim_generator.soundgen import SoundGen

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_target_cfs_from_cochlea():
    """Load target CFs from cochlea data file."""
    cochlea_path = Path(__file__).parent.parent / 'cochlea_meanrate' / 'out' / 'condition_psths' / 'cochlea_psths.mat'
    
    if cochlea_path.exists():
        logger.info(f'Loading target CFs from: {cochlea_path}')
        data = loadmat(str(cochlea_path))
        
        # Extract CFs (convert from cell array if needed)
        cfs = data['cfs']
        if cfs.dtype == object:
            # Cell array of strings
            cfs = np.array([float(cf[0]) for cf in cfs.flatten()])
        else:
            cfs = np.asarray(cfs).flatten()
        
        logger.info(f'Loaded {len(cfs)} target CFs: {cfs.min():.1f} - {cfs.max():.1f} Hz')
        return {'cfs': cfs}
    else:
        # Use synthetic CFs for demonstration
        logger.warning(f'Cochlea file not found at {cochlea_path}')
        logger.info('Using synthetic target CFs for demonstration')
        
        # Create logarithmically spaced CFs similar to cochlea/BEZ
        cfs = np.logspace(np.log10(125), np.log10(4000), 20)
        return {'cfs': cfs}


def example_basic_matching():
    """Example 1: Basic CF matching."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic CF Matching")
    print("=" * 70)
    
    # Load target CFs
    params = load_target_cfs_from_cochlea()
    
    # Create WSR model
    wsr_params = [8, 8, -1, 0]
    wsr_model = WSRModel(wsr_params)
    
    # Create CF matcher
    fs = 16000
    matcher = create_cf_matcher_from_models(wsr_model, params, fs)
    
    # Print mapping summary
    print("\n" + matcher.get_mapping_summary())
    
    # Validate mapping
    print("\nValidating mapping...")
    is_valid = matcher.validate_mapping(tolerance_percent=15.0)
    
    if is_valid:
        print("✓ All mappings are within acceptable tolerance")
    else:
        print("⚠ Some mappings exceed tolerance threshold")
    
    return matcher, wsr_model, params


def example_extract_channels():
    """Example 2: Extract WSR channels at specific CFs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Extract WSR Channels")
    print("=" * 70)
    
    # Create matcher and model from Example 1
    matcher, wsr_model, params = example_basic_matching()
    
    # Generate a test sound
    soundgen = SoundGen(16000, tau=0.005)
    tone = soundgen.sound_maker(
        freq=500, 
        num_harmonics=1, 
        tone_duration=0.2, 
        harmonic_factor=0.5
    )
    
    # Process with WSR model (all 128 channels)
    fs = 16000
    wsr_full_response = wsr_model.process(tone, fs)
    print(f"\nWSR full response shape: {wsr_full_response.shape}")
    print(f"  ({wsr_full_response.shape[0]} time samples × {wsr_full_response.shape[1]} channels)")
    
    # Extract only channels matching target CFs
    extracted_channels, corresponding_cfs = matcher.extract_channels(wsr_full_response)
    print(f"\nExtracted channels shape: {extracted_channels.shape}")
    print(f"  ({extracted_channels.shape[0]} time samples × {extracted_channels.shape[1]} CFs)")
    print(f"\nCorresponding CFs: {corresponding_cfs[:5]}... (showing first 5)")
    
    return wsr_full_response, extracted_channels, corresponding_cfs, matcher


def example_compute_temporal_mean():
    """Example 3: Compute temporal mean for comparison with cochlea/BEZ."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Temporal Mean at Matched CFs")
    print("=" * 70)
    
    # Get extracted channels from Example 2
    _, extracted_channels, corresponding_cfs, matcher = example_extract_channels()
    
    # Compute temporal mean for each CF
    temporal_means = np.mean(extracted_channels, axis=0)
    
    print(f"\nTemporal mean firing rates at each CF:")
    print(f"  Shape: {temporal_means.shape} (one value per CF)")
    
    # Show first few values
    print(f"\nSample values:")
    for i in range(min(5, len(corresponding_cfs))):
        cf = corresponding_cfs[i]
        mean_rate = temporal_means[i]
        wsr_channel = matcher.get_wsr_channel(cf)
        wsr_cf = matcher.wsr_cfs[wsr_channel]
        print(f"  CF={cf:7.1f} Hz (WSR ch {wsr_channel:3d}, CF={wsr_cf:7.1f} Hz): "
              f"mean rate = {mean_rate:.3f}")
    
    return temporal_means, corresponding_cfs


def example_visualization():
    """Example 4: Visualize CF mapping."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Visualize CF Mapping")
    print("=" * 70)
    
    # Load target CFs
    params = load_target_cfs_from_cochlea()
    
    # Create WSR model and matcher
    wsr_params = [8, 8, -1, 0]
    wsr_model = WSRModel(wsr_params)
    fs = 16000
    matcher = create_cf_matcher_from_models(wsr_model, params, fs)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: WSR CFs and target CFs
    ax1 = axes[0]
    ax1.plot(matcher.wsr_cfs, 'b-', label='WSR CFs', alpha=0.5)
    
    # Mark target CFs and their matched WSR channels
    for target_cf in matcher.target_cfs:
        wsr_idx = matcher.get_wsr_channel(target_cf)
        wsr_cf = matcher.wsr_cfs[wsr_idx]
        ax1.plot(wsr_idx, wsr_cf, 'ro', markersize=8, alpha=0.7)
        ax1.axvline(wsr_idx, color='red', alpha=0.2, linestyle='--', linewidth=0.5)
    
    ax1.set_xlabel('WSR Channel Index', fontsize=12)
    ax1.set_ylabel('Center Frequency (Hz)', fontsize=12)
    ax1.set_title('WSR Filterbank CFs with Target CF Mapping', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Mapping accuracy
    ax2 = axes[1]
    target_cfs_sorted = np.sort(matcher.target_cfs)
    percent_diffs = []
    
    for target_cf in target_cfs_sorted:
        wsr_idx = matcher.get_wsr_channel(target_cf)
        wsr_cf = matcher.wsr_cfs[wsr_idx]
        percent_diff = abs(wsr_cf - target_cf) / target_cf * 100
        percent_diffs.append(percent_diff)
    
    ax2.plot(target_cfs_sorted, percent_diffs, 'go-', linewidth=2, markersize=6)
    ax2.axhline(10, color='orange', linestyle='--', linewidth=2, label='10% threshold')
    ax2.axhline(15, color='red', linestyle='--', linewidth=2, label='15% threshold')
    ax2.set_xlabel('Target CF (Hz)', fontsize=12)
    ax2.set_ylabel('Matching Error (%)', fontsize=12)
    ax2.set_title('CF Matching Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / 'cf_mapping_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    
    plt.show()


def example_compare_with_cochlea_structure():
    """Example 5: Create WSR output in same structure as cochlea/BEZ."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: WSR Output in Cochlea/BEZ Structure")
    print("=" * 70)
    
    # Load target parameters
    params = load_target_cfs_from_cochlea()
    target_cfs = params['cfs']
    
    # Create WSR model and matcher
    wsr_params = [8, 8, -1, 0]
    wsr_model = WSRModel(wsr_params)
    fs = 16000
    matcher = create_cf_matcher_from_models(wsr_model, params, fs)
    
    # Simulate multiple frequencies and dB levels (like cochlea/BEZ experiments)
    frequencies = [250, 500, 1000, 2000]
    db_levels = [50, 60, 70, 80]
    
    # Create output structure
    wsr_mean_rates = np.zeros((len(target_cfs), len(frequencies), len(db_levels)))
    
    soundgen = SoundGen(fs, tau=0.005)
    
    print("\nProcessing stimuli...")
    for freq_idx, freq in enumerate(frequencies):
        for db_idx, db in enumerate(db_levels):
            # Generate tone at specific frequency and level
            # (In real use, you'd scale amplitude based on dB level)
            tone = soundgen.sound_maker(
                freq=freq,
                num_harmonics=1,
                tone_duration=0.2,
                harmonic_factor=0.5
            )
            
            # Process with WSR
            wsr_response = wsr_model.process(tone, fs)
            
            # Extract channels matching target CFs
            extracted, _ = matcher.extract_channels(wsr_response)
            
            # Compute temporal mean
            temporal_mean = np.mean(extracted, axis=0)
            
            # Store in structure (CF × freq × dB)
            wsr_mean_rates[:, freq_idx, db_idx] = temporal_mean
            
        print(f"  Processed frequency {freq} Hz across {len(db_levels)} dB levels")
    
    print(f"\n✓ WSR mean rates structure: {wsr_mean_rates.shape}")
    print(f"  Shape: (n_CFs={len(target_cfs)}, n_frequencies={len(frequencies)}, n_dB={len(db_levels)})")
    print(f"  This matches the cochlea/BEZ data structure!")
    
    # Show sample values
    print(f"\nSample values at CF={target_cfs[0]:.1f} Hz:")
    print(f"  Frequencies (Hz): {frequencies}")
    for db_idx, db in enumerate(db_levels):
        rates = wsr_mean_rates[0, :, db_idx]
        print(f"  {db} dB: {rates}")
    
    return wsr_mean_rates


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CF MATCHER USAGE EXAMPLES")
    print("=" * 70)
    
    # Run examples
    example_basic_matching()
    example_extract_channels()
    example_compute_temporal_mean()
    example_visualization()
    example_compare_with_cochlea_structure()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
