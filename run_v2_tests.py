#!/usr/bin/env python3
"""
Run comprehensive V2 tests manually since pytest discovery has issues.
"""

import sys
import traceback

sys.path.insert(0, ".")

from test_v2_comprehensive import TestV2Architecture


def run_comprehensive_tests():
    """Run all V2 tests manually."""
    print("🧪 V2 Architecture Comprehensive Manual Test Run")
    print("=" * 60)

    test_suite = TestV2Architecture()

    tests = [
        ("NumPy Array Compatibility", test_suite.test_numpy_array_compatibility),
        ("Pandas DataFrame Compatibility", test_suite.test_pandas_compatibility),
        ("Edge Cases", test_suite.test_edge_cases),
        ("Column Naming Consistency", test_suite.test_column_naming_consistency),
        ("Config System Integration", test_suite.test_config_system_integration),
        ("Error Handling", test_suite.test_error_handling),
        ("Sklearn Compatibility", test_suite.test_sklearn_compatibility),
    ]

    results = []

    for test_name, test_method in tests:
        print(f"\n🔍 Testing {test_name}")
        print("-" * 50)

        try:
            test_method()
            print(f"✅ {test_name}: PASSED")
            results.append((test_name, True, ""))
        except Exception as e:
            print(f"❌ {test_name}: FAILED")
            print(f"   Error: {str(e)}")
            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Print summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for test_name, success, error in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if not success and error:
            print(f"   Error: {error}")

    print(f"\nTotal: {len(results)} test suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("V2 architecture is working correctly with:")
        print("- Column naming consistency fixed")
        print("- Config system working properly")
        print("- NaN/inf handling robust")
        print("- Transform/inverse transform cycle working")
        print("- Error handling appropriate")
        print("- sklearn compatibility maintained")
    else:
        print(f"\n⚠️  {failed} test suite(s) failed. Please review the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)
