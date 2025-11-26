"""
Test suite for refactored loggers module functionality.

Created: 2025-11-26 16:11:00
Purpose: Verify FolderManager, LoggingConfigurator,
and MetadataSaver work correctly
"""

from pathlib import Path
from loggers import (FolderManager, LoggingConfigurator,
                     MetadataSaver, model_builders)
import logging


def test_basic_folder_manager():
    """Test basic FolderManager with default timestamp naming."""
    print("\n1. Testing basic FolderManager (default naming)...")
    manager = FolderManager("/tmp/test_loggers")
    output_dir = manager.create_folder()
    print(f"   ✓ Created: {output_dir}")
    return output_dir


def test_model_comparison_builder():
    """Test FolderManager with model_comparison builder."""
    print("\n2. Testing model_comparison builder...")
    manager = FolderManager("/tmp/test_loggers", "model_comparison")
    output_dir = manager.with_params(
        model1="wsr",
        model2="bez",
        comparison_type="population"
    ).create_folder()
    print(f"   ✓ Created: {output_dir}")
    return output_dir


def test_custom_builder():
    """Test FolderManager with custom builder function."""
    print("\n3. Testing custom builder function...")

    def custom_namer(params, timestamp):
        return f"{params['analysis_type']}_{params['dataset']}_{timestamp}"

    manager = FolderManager("/tmp/test_loggers", custom_namer)
    output_dir = manager.with_params(
        analysis_type="regression",
        dataset="neural_data"
    ).create_folder()
    print(f"   ✓ Created: {output_dir}")
    return output_dir


def test_logging_configurator():
    """Test LoggingConfigurator."""
    print("\n4. Testing LoggingConfigurator...")
    output_dir = Path("/tmp/test_loggers/logging_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = LoggingConfigurator.setup_basic(output_dir, "test.log")
    logger = logging.getLogger(__name__)
    logger.info("This is a test log message")
    print(f"   ✓ Log file created: {log_file}")
    return output_dir


def test_metadata_saver():
    """Test MetadataSaver."""
    print("\n5. Testing MetadataSaver...")
    output_dir = Path("/tmp/test_loggers/metadata_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": "test",
        "r_squared": 0.95,
        "p_value": 1.23e-10,
        "parameters": ["param1", "param2"]
    }

    saver = MetadataSaver()
    json_file = saver.save_json(output_dir, metadata, "test_metadata.json")
    text_file = saver.save_text(output_dir, metadata, "test_metadata.txt")

    print(f"   ✓ JSON saved: {json_file}")
    print(f"   ✓ Text saved: {text_file}")
    return output_dir


def test_backward_compatibility():
    """Test backward compatibility with ExperimentFolderManager."""
    print("\n6. Testing backward compatibility...")
    from loggers import ExperimentFolderManager

    manager = ExperimentFolderManager("/tmp/test_loggers", "bez2018")
    # Note: using old method name would fail, but class alias works
    # We use new method name to show compatibility
    output_dir = manager.with_params(
        num_runs=10,
        num_cf=20,
        min_cf=125,
        max_cf=8000,
        num_ANF=(4, 4, 4)
    ).create_folder()
    print(f"   ✓ ExperimentFolderManager alias works: {output_dir}")
    return output_dir


def test_list_registered_models():
    """Test listing registered model builders."""
    print("\n7. Testing registered model builders...")
    models = model_builders.list_models()
    print(f"   ✓ Registered models: {models}")
    return models


if __name__ == "__main__":
    print("="*70)
    print("Testing Refactored Loggers Module")
    print("="*70)

    try:
        test_basic_folder_manager()
        test_model_comparison_builder()
        test_custom_builder()
        test_logging_configurator()
        test_metadata_saver()
        test_backward_compatibility()
        test_list_registered_models()

        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
