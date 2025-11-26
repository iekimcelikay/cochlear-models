"""
CF Matcher Module

This module handles matching between WSR model channel frequencies and the
characteristic frequencies (CFs) used in cochlea and BEZ2018 auditory nerve models.

The WSR model uses a fixed filterbank with 128 channels, while cochlea/BEZ models
use a specific set of CFs. This module maps between the two representations.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CFMatcher:
    """
    Maps WSR model channel frequencies to cochlea/BEZ characteristic frequencies.
    
    Attributes:
        wsr_cfs (np.ndarray): WSR filterbank center frequencies
        target_cfs (np.ndarray): Target CFs from cochlea/BEZ models
        mapping (Dict[float, int]): Maps target CF to nearest WSR channel index
        reverse_mapping (Dict[int, float]): Maps WSR channel to nearest target CF
    """
    
    def __init__(self, wsr_cfs: np.ndarray, target_cfs: np.ndarray):
        """
        Initialize the CF matcher.
        
        Args:
            wsr_cfs: Array of WSR filterbank center frequencies (Hz)
            target_cfs: Array of target CFs from cochlea/BEZ (Hz)
        """
        self.wsr_cfs = np.asarray(wsr_cfs)
        self.target_cfs = np.asarray(target_cfs)
        
        # Create bidirectional mapping
        self.mapping = self._create_mapping()
        self.reverse_mapping = self._create_reverse_mapping()
        
        logger.info(
            f'CFMatcher initialized: {len(self.wsr_cfs)} WSR channels, '
            f'{len(self.target_cfs)} target CFs'
        )
        
    def _create_mapping(self) -> Dict[float, int]:
        """
        Create mapping from target CFs to nearest WSR channel indices.
        
        Returns:
            Dictionary mapping each target CF to its nearest WSR channel index
        """
        mapping = {}
        
        for target_cf in self.target_cfs:
            # Find nearest WSR channel (Euclidean distance in log space for better frequency matching)
            log_distances = np.abs(np.log10(self.wsr_cfs) - np.log10(target_cf))
            nearest_idx = np.argmin(log_distances)
            mapping[float(target_cf)] = int(nearest_idx)
            
            # Log if there's a large mismatch
            wsr_cf = self.wsr_cfs[nearest_idx]
            percent_diff = abs(wsr_cf - target_cf) / target_cf * 100
            
            if percent_diff > 10:
                logger.warning(
                    f'Large CF mismatch: target={target_cf:.1f} Hz -> '
                    f'WSR channel {nearest_idx} (CF={wsr_cf:.1f} Hz, '
                    f'{percent_diff:.1f}% difference)'
                )
        
        return mapping
    
    def _create_reverse_mapping(self) -> Dict[int, float]:
        """
        Create reverse mapping from WSR channel indices to nearest target CFs.
        
        Returns:
            Dictionary mapping WSR channel indices to their nearest target CF
        """
        reverse_mapping = {}
        
        for wsr_idx, wsr_cf in enumerate(self.wsr_cfs):
            # Find nearest target CF
            distances = np.abs(self.target_cfs - wsr_cf)
            nearest_idx = np.argmin(distances)
            reverse_mapping[int(wsr_idx)] = float(self.target_cfs[nearest_idx])
        
        return reverse_mapping
    
    def get_wsr_channel(self, target_cf: float) -> int:
        """
        Get WSR channel index for a given target CF.
        
        Args:
            target_cf: Target characteristic frequency (Hz)
            
        Returns:
            WSR channel index (0-127 for standard filterbank)
            
        Raises:
            KeyError: If target_cf is not in the mapping
        """
        return self.mapping[float(target_cf)]
    
    def get_target_cf(self, wsr_channel: int) -> float:
        """
        Get nearest target CF for a given WSR channel index.
        
        Args:
            wsr_channel: WSR channel index
            
        Returns:
            Nearest target CF (Hz)
            
        Raises:
            KeyError: If wsr_channel is not in the reverse mapping
        """
        return self.reverse_mapping[int(wsr_channel)]
    
    def extract_channels(
        self, 
        wsr_response: np.ndarray, 
        target_cfs: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract WSR channels corresponding to target CFs.
        
        Args:
            wsr_response: WSR model response array (n_time, n_channels)
            target_cfs: Optional list of specific target CFs to extract.
                       If None, uses all target CFs from initialization.
                       
        Returns:
            Tuple of (extracted_channels, corresponding_cfs):
                - extracted_channels: Array (n_time, n_target_cfs)
                - corresponding_cfs: Array of target CFs used
        """
        if target_cfs is None:
            target_cfs = self.target_cfs
        else:
            target_cfs = np.asarray(target_cfs)
        
        # Get channel indices
        channel_indices = [self.get_wsr_channel(cf) for cf in target_cfs]
        
        # Extract channels
        extracted = wsr_response[:, channel_indices]
        
        logger.info(
            f'Extracted {len(channel_indices)} channels from WSR response '
            f'(shape: {wsr_response.shape} -> {extracted.shape})'
        )
        
        return extracted, target_cfs
    
    def get_wsr_channel_indices(self) -> np.ndarray:
        """
        Get array of WSR channel indices that match to target CFs.
        
        Returns:
            Array of WSR channel indices in same order as target_cfs
        """
        indices = np.array([self.mapping[cf] for cf in sorted(self.target_cfs)])
        return indices
    
    def get_mapping_table(self) -> List[Dict[str, float]]:
        """
        Get complete mapping table as list of dictionaries for easy viewing/saving.
        
        Returns:
            List of dicts with keys: 'target_cf', 'wsr_channel', 'wsr_cf', 
            'diff_hz', 'diff_percent'
        """
        table = []
        
        for target_cf in sorted(self.target_cfs):
            wsr_idx = self.mapping[target_cf]
            wsr_cf = self.wsr_cfs[wsr_idx]
            diff_hz = abs(wsr_cf - target_cf)
            diff_percent = (diff_hz / target_cf) * 100
            
            table.append({
                'target_cf': float(target_cf),
                'wsr_channel': int(wsr_idx),
                'wsr_cf': float(wsr_cf),
                'diff_hz': float(diff_hz),
                'diff_percent': float(diff_percent)
            })
        
        return table
    
    def save_mapping_table(self, filepath: Path) -> None:
        """
        Save mapping table to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        import csv
        
        table = self.get_mapping_table()
        
        with open(filepath, 'w', newline='') as f:
            if table:
                writer = csv.DictWriter(f, fieldnames=table[0].keys())
                writer.writeheader()
                writer.writerows(table)
        
        logger.info(f'Saved mapping table to {filepath}')
    
    def get_mapping_summary(self) -> str:
        """
        Get a summary of the CF mapping.
        
        Returns:
            Formatted string describing the mapping
        """
        lines = [
            "CF Mapping Summary",
            "="*70,
            f"WSR Channels: {len(self.wsr_cfs)} (range: {self.wsr_cfs.min():.1f} - {self.wsr_cfs.max():.1f} Hz)",
            f"Target CFs: {len(self.target_cfs)} (range: {self.target_cfs.min():.1f} - {self.target_cfs.max():.1f} Hz)",
            "",
            "Sample Mappings (first 5 target CFs):",
            "-"*70
        ]
        
        for target_cf in sorted(self.target_cfs)[:5]:
            wsr_idx = self.mapping[target_cf]
            wsr_cf = self.wsr_cfs[wsr_idx]
            diff = abs(wsr_cf - target_cf)
            percent_diff = diff / target_cf * 100
            
            lines.append(
                f"  Target CF {target_cf:7.1f} Hz -> WSR channel {wsr_idx:3d} "
                f"(CF={wsr_cf:7.1f} Hz, diff={percent_diff:5.2f}%)"
            )
        
        if len(self.target_cfs) > 5:
            lines.append("  ...")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def validate_mapping(self, tolerance_percent: float = 15.0) -> bool:
        """
        Validate that all mappings are within acceptable tolerance.
        
        Args:
            tolerance_percent: Maximum acceptable percentage difference
            
        Returns:
            True if all mappings are within tolerance, False otherwise
        """
        all_valid = True
        
        for target_cf in self.target_cfs:
            wsr_idx = self.mapping[target_cf]
            wsr_cf = self.wsr_cfs[wsr_idx]
            percent_diff = abs(wsr_cf - target_cf) / target_cf * 100
            
            if percent_diff > tolerance_percent:
                logger.error(
                    f'Mapping exceeds tolerance: target={target_cf:.1f} Hz -> '
                    f'WSR CF={wsr_cf:.1f} Hz ({percent_diff:.1f}% > {tolerance_percent}%)'
                )
                all_valid = False
        
        if all_valid:
            logger.info(
                f'All CF mappings within {tolerance_percent}% tolerance'
            )
        
        return all_valid


def create_cf_matcher_from_models(
    wsr_model,
    params: Dict,
    fs: int = 16000
) -> CFMatcher:
    """
    Create a CFMatcher from a WSR model and cochlea/BEZ parameters.
    
    Args:
        wsr_model: WSRModel instance with get_filterbank_CFs method
        params: Dictionary containing 'cfs' key with target CFs
        fs: Sampling frequency (Hz)
        
    Returns:
        Configured CFMatcher instance
        
    Example:
        >>> from wsrmodel import WSRModel
        >>> from scipy.io import loadmat
        >>> 
        >>> # Load target CFs from cochlea/BEZ data
        >>> data = loadmat('cochlea_psths.mat')
        >>> params = {'cfs': data['cfs']}
        >>> 
        >>> # Create WSR model
        >>> wsr_params = [8, 8, -1, 0]
        >>> wsr_model = WSRModel(wsr_params)
        >>> 
        >>> # Create matcher
        >>> matcher = create_cf_matcher_from_models(wsr_model, params, fs=16000)
        >>> print(matcher.get_mapping_summary())
    """
    # Get WSR filterbank CFs
    wsr_cfs = wsr_model.get_filterbank_CFs(fs)
    
    # Get target CFs from params
    target_cfs = np.asarray(params['cfs'])
    
    # Create and return matcher
    matcher = CFMatcher(wsr_cfs, target_cfs)
    
    return matcher


def create_cf_matcher_from_bez2018(
    wsr_model,
    freq_range: Tuple[float, float, int],
    fs: int = 16000,
    species: str = 'human',
    min_cf: float = 180.0
) -> CFMatcher:
    """
    Create a CFMatcher using BEZ2018's calc_cfs function to generate target CFs.
    
    Filters out CFs below min_cf to match WSR filterbank range (starts at ~180 Hz).
    Also uses WSR channels [1:129] (skipping channel 0).
    
    Args:
        wsr_model: WSRModel instance with get_filterbank_CFs method
        freq_range: Tuple of (min_cf, max_cf, num_cf) for BEZ2018 model
        fs: Sampling frequency for WSR model (Hz)
        species: Species for Greenwood function ('human' or 'cat')
        min_cf: Minimum CF to include (default: 180.0 Hz, WSR filterbank lower limit)
        
    Returns:
        Configured CFMatcher instance
        
    Example:
        >>> from wsrmodel import WSRModel
        >>> 
        >>> # Create WSR model
        >>> wsr_model = WSRModel([8, 8, -1, 0])
        >>> 
        >>> # Create matcher with same CFs as BEZ2018 (filtered to WSR range)
        >>> matcher = create_cf_matcher_from_bez2018(
        ...     wsr_model,
        ...     freq_range=(125, 2500, 20),  # Same as BEZ2018 setup
        ...     fs=16000
        ... )
        >>> print(matcher.get_mapping_summary())
    """
    # Import calc_cfs from BEZ2018 utilities
    from BEZ2018_meanrate.simulator_utils import calc_cfs
    
    # Get WSR filterbank CFs (skip first channel, use [1:129])
    all_wsr_cfs = wsr_model.get_filterbank_CFs(fs)
    wsr_cfs = all_wsr_cfs[1:129]  # Use channels 1-128, skip channel 0
    
    # Get BEZ2018 CFs using calc_cfs
    all_bez_cfs = calc_cfs(freq_range, species=species)
    
    # Filter BEZ CFs to only include those >= min_cf (to match WSR range)
    bez_cfs = all_bez_cfs[all_bez_cfs >= min_cf]
    
    # Create and return matcher
    matcher = CFMatcher(wsr_cfs, bez_cfs)
    
    n_skipped = len(all_bez_cfs) - len(bez_cfs)
    logger.info(
        f'Created CF matcher from BEZ2018 config: {len(bez_cfs)} CFs '
        f'({bez_cfs.min():.1f} - {bez_cfs.max():.1f} Hz), '
        f'skipped {n_skipped} CFs below {min_cf} Hz'
    )
    logger.info(
        f'WSR channels: {len(wsr_cfs)} (indices 1-128, skipped channel 0)'
    )
    
    return matcher


# Example usage / testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("CF Matcher Demo: BEZ2018 CFs matching to WSR channels")
    print("=" * 70)
    
    # Create WSR model
    from wsrmodel import WSRModel
    wsr_model = WSRModel([8, 8, -1, 0])
    
    # Create matcher with same CFs as BEZ2018
    matcher = create_cf_matcher_from_bez2018(
        wsr_model,
        freq_range=(125, 2500, 20),  # Same as BEZ2018 setup
        fs=16000
    )
    
    # Print mapping summary
    print("\n" + matcher.get_mapping_summary())
    
    # Save all WSR channel CFs to file
    wsr_channels_file = Path('wsr_all_channels.csv')
    import csv
    with open(wsr_channels_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['channel_index', 'cf_hz', 'matched_to_bez'])
        for idx, cf in enumerate(matcher.wsr_cfs):
            is_matched = 'yes' if idx in matcher.mapping.values() else 'no'
            writer.writerow([idx, f'{cf:.2f}', is_matched])
    
    print(f"\n✓ All {len(matcher.wsr_cfs)} WSR channels saved to: {wsr_channels_file}")
    
    # Get WSR channel indices
    wsr_indices = matcher.get_wsr_channel_indices()
    print("\nWSR Channel Indices (matched to BEZ CFs):")
    print(wsr_indices)
    
    # Get mapping table
    mapping_table = matcher.get_mapping_table()
    
    # Print detailed mapping for all CFs
    print("\n" + "="*70)
    print("COMPLETE CF MAPPING TABLE")
    print("="*70)
    print(f"{'BEZ CF (Hz)':>12} | {'WSR Channel':>12} | {'WSR CF (Hz)':>12} | {'Diff (Hz)':>10} | {'Diff (%)':>8}")
    print("-"*70)
    
    for row in mapping_table:
        print(f"{row['target_cf']:12.2f} | {row['wsr_channel']:12d} | "
              f"{row['wsr_cf']:12.2f} | {row['diff_hz']:10.2f} | {row['diff_percent']:7.2f}%")
    
    # Save mapping table to CSV
    output_file = Path('wsr_bez_cf_mapping.csv')
    matcher.save_mapping_table(output_file)
    print(f"\n✓ Mapping table saved to: {output_file}")
    
    # Validate mapping
    print("\n" + "=" * 70)
    print("Validating CF mappings (15% tolerance)...")
    is_valid = matcher.validate_mapping(tolerance_percent=15.0)
    
    if is_valid:
        print("✓ All CF mappings are valid!")
    else:
        print("✗ Some CF mappings exceed tolerance")
    
    print("\n" + "=" * 70)