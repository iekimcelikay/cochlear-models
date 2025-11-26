#!/usr/bin/env python3
"""
Quick test to verify the hybrid regression implementation works correctly
Run this to ensure all functions are accessible and have correct signatures
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all functions can be imported"""
    print("Testing imports...")
    
    from subcorticalSTRF.compare_bez_cochlea_mainscript import (
        # Level 1: Core utilities
        values_match,
        find_matching_value,
        compute_single_regression,
        extract_mean_rates_from_nested_dict,
        
        # Level 2: Data matching
        match_data_points,
        
        # Level 4: Public API - Single comparisons
        regress_cochlea_vs_bez_single_run,
        regress_bez_run_vs_run,
        
        # Level 4: Public API - Batch comparisons
        regress_cochlea_vs_bez_all_runs,
        regress_all_bez_run_pairs,
        
        # Level 4: Public API - Grouped analysis
        regress_by_fiber_type,
        regress_by_cf,
        regress_by_db,
        correlate_by_fiber_type,
    )
    
    print("  ✓ All imports successful!")
    return True


def test_values_match():
    """Test fuzzy matching utility"""
    print("\nTesting values_match()...")
    
    from subcorticalSTRF.compare_bez_cochlea_mainscript import values_match
    
    # Test exact match
    assert values_match(125.0, 125.0), "Exact match should pass"
    
    # Test fuzzy match (within tolerance)
    assert values_match(125.0, 125.000001, rtol=1e-5), "Fuzzy match should pass"
    
    # Test mismatch (outside tolerance)
    assert not values_match(125.0, 126.0, rtol=1e-5), "Large difference should fail"
    
    print("  ✓ values_match() works correctly!")
    return True


def test_compute_single_regression():
    """Test regression computation"""
    print("\nTesting compute_single_regression()...")
    
    import numpy as np
    from subcorticalSTRF.compare_bez_cochlea_mainscript import compute_single_regression
    
    # Create simple test data: y = 2x + 1 with noise
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 0.1, 100)
    
    result = compute_single_regression(x, y, condition_info={'test': 'data'})
    
    # Check return type (returns dict, not DataFrame)
    assert isinstance(result, dict), "Should return dict"
    
    # Check keys
    expected_keys = ['slope', 'intercept', 'r_value', 'p_value', 'std_err', 'n_points']
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check values are reasonable
    slope = result['slope']
    intercept = result['intercept']
    r_value = result['r_value']
    
    assert 1.9 < slope < 2.1, f"Slope should be ~2, got {slope}"
    assert 0.9 < intercept < 1.1, f"Intercept should be ~1, got {intercept}"
    assert r_value > 0.99, f"Correlation should be >0.99, got {r_value}"
    
    print(f"  ✓ Regression: slope={slope:.3f}, intercept={intercept:.3f}, r={r_value:.3f}")
    return True


def test_mock_workflow():
    """Test workflow with mock data"""
    print("\nTesting workflow with mock data...")
    
    import numpy as np
    from subcorticalSTRF.compare_bez_cochlea_mainscript import match_data_points, regress_by_fiber_type
    
    # Create mock nested dicts (simplified structure)
    mock_data1 = {
        'hsr': {
            125.0: {
                125.0: {
                    30.0: 10.5  # Single value (no runs)
                }
            }
        },
        'msr': {
            125.0: {
                125.0: {
                    30.0: 8.2
                }
            }
        }
    }
    
    mock_data2 = {
        'hsr': {
            125.0: {
                125.0: {
                    30.0: {
                        0: 10.3,  # Run 0
                        1: 10.7   # Run 1
                    }
                }
            }
        },
        'msr': {
            125.0: {
                125.0: {
                    30.0: {
                        0: 8.1,
                        1: 8.3
                    }
                }
            }
        }
    }
    
    # Test match_data_points
    matched_rates, matched_meta = match_data_points(
        mock_data1, mock_data2,
        fiber_type='hsr',
        run2_idx=0,
        verbose=False
    )
    
    assert len(matched_rates) == 1, f"Should match 1 point, got {len(matched_rates)}"
    assert abs(matched_rates[0]['data1_rate'] - 10.5) < 0.01, "data1_rate mismatch"
    assert abs(matched_rates[0]['data2_rate'] - 10.3) < 0.01, "data2_rate mismatch"
    
    print("  ✓ Mock workflow successful!")
    print(f"    Matched: {len(matched_rates)} points")
    print(f"    Data1: {matched_rates[0]['data1_rate']}, Data2: {matched_rates[0]['data2_rate']}")
    
    return True


def test_function_signatures():
    """Test that functions have expected parameters"""
    print("\nTesting function signatures...")
    
    import inspect
    from subcorticalSTRF.compare_bez_cochlea_mainscript import (
        regress_cochlea_vs_bez_single_run,
        regress_by_fiber_type,
        match_data_points
    )
    
    # Check regress_cochlea_vs_bez_single_run
    sig = inspect.signature(regress_cochlea_vs_bez_single_run)
    params = list(sig.parameters.keys())
    assert 'cochlea_means' in params, "Missing cochlea_means parameter"
    assert 'bez_means' in params, "Missing bez_means parameter"
    assert 'run_idx' in params, "Missing run_idx parameter"
    assert 'rtol' in params, "Missing rtol parameter"
    assert 'atol' in params, "Missing atol parameter"
    
    # Check regress_by_fiber_type
    sig = inspect.signature(regress_by_fiber_type)
    params = list(sig.parameters.keys())
    assert 'data1_means' in params, "Missing data1_means parameter"
    assert 'data2_means' in params, "Missing data2_means parameter"
    assert 'run1_idx' in params, "Missing run1_idx parameter"
    assert 'run2_idx' in params, "Missing run2_idx parameter"
    
    # Check match_data_points
    sig = inspect.signature(match_data_points)
    params = list(sig.parameters.keys())
    assert 'data1_means' in params, "Missing data1_means parameter"
    assert 'data2_means' in params, "Missing data2_means parameter"
    assert 'fiber_type' in params, "Missing fiber_type parameter"
    
    print("  ✓ All function signatures correct!")
    return True


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("TESTING HYBRID REGRESSION IMPLEMENTATION")
    print("="*60)
    
    tests = [
        test_imports,
        test_values_match,
        test_compute_single_regression,
        test_function_signatures,
        test_mock_workflow,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n🎉 All tests passed! The hybrid implementation is ready to use.")
        print("\nNext steps:")
        print("  1. See REGRESSION_USAGE_EXAMPLES.md for usage examples")
        print("  2. Try: python3 -c 'from subcorticalSTRF.data_loader import regress_cochlea_vs_bez_single_run; help(regress_cochlea_vs_bez_single_run)'")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(run_all_tests())
