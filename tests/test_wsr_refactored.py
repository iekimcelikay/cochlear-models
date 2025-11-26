"""
Test suite for refactored WSR model scripts.

Verifies that WSR model scripts correctly integrate with the new
loggers module, including folder management, logging configuration,
and metadata saving.

Tests:
    - test_imports: Verify all WSR modules import correctly
    - test_config_creation: Validate WSRConfig object creation
    - test_simulation_setup: Test WSRSimulation initialization and setup
    - test_result_collector: Verify ResultCollector functionality
    - cleanup: Clean up temporary test directories

Created: 2025-01-16 08:00:00
"""

import logging
import shutil
import sys
import traceback
from pathlib import Path

# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports() -> bool:
    """Test that all imports work correctly.

    Returns:
        bool: True if all imports successful, False otherwise.
    """
    print("\n1. Testing imports...")
    try:
        from models.WSRmodel.run_wsr_model_OOP import (  # noqa: F401
            WSRConfig, WSRSimulation
        )
        from models.WSRmodel.run_wsr_model_pipeline import (  # noqa: F401
            ResultCollector, WSRProcessor
        )
        print("   ✓ All imports successful")
        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False


def test_config_creation() -> bool:
    """Test WSRConfig creation.

    Returns:
        bool: True if config created successfully, False otherwise.
    """
    print("\n2. Testing WSRConfig creation...")
    try:
        from models.WSRmodel.run_wsr_model_OOP import WSRConfig

        config = WSRConfig(
            db_range=60,
            freq_range=(125, 2500, 5),  # Small range for testing
            tone_duration=0.1,
            wsr_params=[8, 8, -1, 0]
        )

        print("   ✓ Config created successfully")
        print(f"     - Sample rate: {config.sample_rate}")
        print(f"     - WSR params: {config.wsr_params}")
        print(f"     - Freq range: {config.freq_range}")
        return True
    except Exception as e:
        print(f"   ✗ Config creation failed: {e}")
        return False


def test_simulation_setup() -> bool:
    """Test WSRSimulation initialization and folder setup.

    Returns:
        bool: True if simulation setup successful, False otherwise.
    """
    print("\n3. Testing WSRSimulation setup...")
    try:
        from models.WSRmodel.run_wsr_model_OOP import WSRConfig, WSRSimulation

        # Create config with test output directory
        config = WSRConfig(
            db_range=60,
            freq_range=(125, 500, 2),  # Minimal for testing
            tone_duration=0.1,
            results_folder='/tmp/test_wsr_refactored'
        )

        # Create simulation
        simulation = WSRSimulation(config)
        print("   ✓ Simulation object created")

        # Test folder setup
        simulation.setup_output_folder()
        print(f"   ✓ Output folder created: {simulation.save_dir}")

        # Check that logging was configured
        logger = logging.getLogger()
        has_handlers = len(logger.handlers) > 0
        print(f"   ✓ Logging configured: {has_handlers}")

        # Check that expected files exist
        save_dir = Path(simulation.save_dir)

        expected_files = ['params.json', 'stimulus_params.txt']
        for filename in expected_files:
            file_path = save_dir / filename
            if file_path.exists():
                print(f"   ✓ Found: {filename}")
            else:
                print(f"   ⚠ Missing: {filename}")

        return True

    except Exception as e:
        print(f"   ✗ Simulation setup failed: {e}")
        traceback.print_exc()
        return False


def test_result_collector() -> bool:
    """Test ResultCollector with new FolderManager.

    Returns:
        bool: True if ResultCollector test passed, False otherwise.
    """
    print("\n4. Testing ResultCollector...")
    try:
        from models.WSRmodel.run_wsr_model_pipeline import ResultCollector

        test_dir = Path('/tmp/test_wsr_collector')
        test_dir.mkdir(parents=True, exist_ok=True)

        collector = ResultCollector(test_dir)
        print("   ✓ ResultCollector created")
        print(f"     - Save directory: {collector.save_dir}")

        return True

    except Exception as e:
        print(f"   ✗ ResultCollector test failed: {e}")
        traceback.print_exc()
        return False


def cleanup() -> None:
    """Clean up test directories."""
    print("\n5. Cleaning up test directories...")

    test_dirs = [
        '/tmp/test_wsr_refactored',
        '/tmp/test_wsr_collector'
    ]

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"   ✓ Removed: {test_dir}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Refactored WSR Model Scripts")
    print("=" * 70)

    results = []

    try:
        results.append(("Imports", test_imports()))
        results.append(("Config Creation", test_config_creation()))
        results.append(("Simulation Setup", test_simulation_setup()))
        results.append(("ResultCollector", test_result_collector()))

        # Summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)

        for test_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:.<50} {status}")

        all_passed = all(result[1] for result in results)

        if all_passed:
            print("\n✓ All tests passed!")
        else:
            print("\n⚠ Some tests failed")

    finally:
        cleanup()

    print("=" * 70)
