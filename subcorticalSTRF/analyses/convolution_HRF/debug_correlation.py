#!/usr/bin/env python3
"""
Debug script for correlation analysis.
"""

import logging
from correlation_analysis import analyze_correlations
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

# File paths (adjust these to your actual files)
bez_file = 'results/extracted/BEZ_extracted_60.0dB_20241028_155115.pkl'
cochlea_file = 'results/extracted/Cochlea_extracted_60.0dB_20241028_174325.pkl'

# Check files exist
bez_path = Path(bez_file)
cochlea_path = Path(cochlea_file)

print(f"BEZ file exists: {bez_path.exists()}")
print(f"Cochlea file exists: {cochlea_path.exists()}")

if bez_path.exists() and cochlea_path.exists():
    # Run analysis
    results = analyze_correlations(
        bez_file=bez_file,
        cochlea_file=cochlea_file,
        fiber_type='hsr'
    )
else:
    print("\nERROR: One or both files not found!")
    print(f"BEZ: {bez_path.absolute()}")
    print(f"Cochlea: {cochlea_path.absolute()}")
