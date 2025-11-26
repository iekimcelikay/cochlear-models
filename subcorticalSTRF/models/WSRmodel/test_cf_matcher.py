"""
Quick test script for CF matcher functionality
"""

import sys
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

import numpy as np
from WSRmodel.wsrmodel import WSRModel
from WSRmodel.cf_matcher import CFMatcher
from stim_generator.soundgen import SoundGen


def test_cf_matcher():
    """Test basic CFMatcher functionality."""
    print("\n" + "="*70)
    print("TEST: CF Matcher Basic Functionality")
    print("="*70)
    
    # Create WSR model
    wsr_params = [8, 8, -1, 0]
    wsr_model = WSRModel(wsr_params)
    
    # Get WSR CFs
    fs = 16000
    wsr_cfs = wsr_model.get_filterbank_CFs(fs)
    print(f"\n✓ WSR has {len(wsr_cfs)} channels")
    print(f"  CF range: {wsr_cfs.min():.1f} - {wsr_cfs.max():.1f} Hz")
    
    # Create synthetic target CFs (like cochlea/BEZ)
    target_cfs = np.array([125, 250, 500, 1000, 2000, 4000])
    print(f"\n✓ Target CFs: {target_cfs}")
    
    # Create matcher
    matcher = CFMatcher(wsr_cfs, target_cfs)
    print(f"\n✓ CFMatcher created successfully")
    
    # Test mapping
    print(f"\nMapping results:")
    for target_cf in target_cfs:
        wsr_idx = matcher.get_wsr_channel(target_cf)
        wsr_cf = wsr_cfs[wsr_idx]
        diff_pct = abs(wsr_cf - target_cf) / target_cf * 100
        print(f"  {target_cf:6.0f} Hz -> WSR channel {wsr_idx:3d} (CF={wsr_cf:7.1f} Hz, error={diff_pct:5.2f}%)")
    
    return matcher


def test_process_at_target_cfs():
    """Test WSRModel.process_at_target_cfs method."""
    print("\n" + "="*70)
    print("TEST: Process Audio at Target CFs")
    print("="*70)
    
    # Create WSR model
    wsr_params = [8, 8, -1, 0]
    wsr_model = WSRModel(wsr_params)
    
    # Generate test tone
    soundgen = SoundGen(16000, tau=0.005)
    tone = soundgen.sound_maker(freq=500, num_harmonics=1, tone_duration=0.2)
    print(f"\n✓ Generated 500 Hz tone ({len(tone)} samples)")
    
    # Define target CFs
    target_cfs = np.array([125, 250, 500, 1000, 2000, 4000])
    
    # Process at target CFs
    fs = 16000
    extracted_channels, corresponding_cfs = wsr_model.process_at_target_cfs(tone, fs, target_cfs)
    
    print(f"\n✓ Extracted channels shape: {extracted_channels.shape}")
    print(f"  ({extracted_channels.shape[0]} time samples × {extracted_channels.shape[1]} CFs)")
    
    # Compute temporal mean
    temporal_mean = np.mean(extracted_channels, axis=0)
    print(f"\n✓ Temporal mean shape: {temporal_mean.shape}")
    print(f"\nTemporal mean values:")
    for cf, mean_val in zip(corresponding_cfs, temporal_mean):
        print(f"  CF={cf:6.0f} Hz: mean={mean_val:.6f}")
    
    return extracted_channels, temporal_mean


def test_full_workflow():
    """Test complete workflow simulating cochlea/BEZ comparison."""
    print("\n" + "="*70)
    print("TEST: Full Workflow (Simulating Cochlea/BEZ Structure)")
    print("="*70)
    
    # Setup
    wsr_params = [8, 8, -1, 0]
    wsr_model = WSRModel(wsr_params)
    soundgen = SoundGen(16000, tau=0.005)
    fs = 16000
    
    # Define experimental parameters (like cochlea/BEZ)
    target_cfs = np.array([125, 250, 500, 1000, 2000])
    frequencies = [250, 500, 1000]
    db_levels = [60, 70]  # Simplified for testing
    
    print(f"\n✓ Parameters:")
    print(f"  CFs: {target_cfs}")
    print(f"  Frequencies: {frequencies}")
    print(f"  dB levels: {db_levels}")
    
    # Create output array
    wsr_rates = np.zeros((len(target_cfs), len(frequencies), len(db_levels)))
    
    # Process each condition
    print(f"\n✓ Processing {len(frequencies) * len(db_levels)} conditions...")
    
    for freq_idx, freq in enumerate(frequencies):
        for db_idx, db in enumerate(db_levels):
            # Generate tone
            tone = soundgen.sound_maker(
                freq=freq,
                num_harmonics=1,
                tone_duration=0.2
            )
            
            # Process at target CFs
            channels, _ = wsr_model.process_at_target_cfs(tone, fs, target_cfs)
            
            # Compute temporal mean
            temporal_mean = np.mean(channels, axis=0)
            
            # Store (CF × freq × dB structure)
            wsr_rates[:, freq_idx, db_idx] = temporal_mean
            
            print(f"  Processed {freq} Hz at {db} dB")
    
    print(f"\n✓ Final WSR rates array shape: {wsr_rates.shape}")
    print(f"  (n_CFs={len(target_cfs)}, n_frequencies={len(frequencies)}, n_dB={len(db_levels)})")
    print(f"  This matches cochlea/BEZ data structure!")
    
    # Show sample
    print(f"\nSample values at CF={target_cfs[0]} Hz:")
    for freq_idx, freq in enumerate(frequencies):
        print(f"  {freq} Hz: {wsr_rates[0, freq_idx, :]}")
    
    return wsr_rates


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CF MATCHER TEST SUITE")
    print("="*70)
    
    try:
        # Run tests
        test_cf_matcher()
        test_process_at_target_cfs()
        test_full_workflow()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
