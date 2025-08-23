"""
Test suite for ATES validation framework 

This module provides unit tests and integration tests for the validation framework
to ensure reliability and correctness of the validation process.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import the validation framework
from validation.validation_framework import (
    ATESValidationFramework, 
    ValidationCase, 
    ValidationMetrics, 
    create_predefined_validation_cases,
    run_comprehensive_validation_suite
)


class TestValidationMetrics(unittest.TestCase):
    """Test cases for ValidationMetrics class."""
    
    def test_calculate_percentage_difference(self):
        """Test percentage difference calculation."""
        # Normal case
        self.assertAlmostEqual(
            ValidationMetrics.calculate_percentage_difference(10.0, 9.0), 
            11.11, places=2
        )
        
        # Zero expected value
        self.assertEqual(
            ValidationMetrics.calculate_percentage_difference(5.0, 0.0), 
            float('inf')
        )
        
        # Both zero
        self.assertEqual(
            ValidationMetrics.calculate_percentage_difference(0.0, 0.0), 
            0.0
        )
        
        # Negative values
        self.assertAlmostEqual(
            ValidationMetrics.calculate_percentage_difference(-5.0, -4.0), 
            25.0, places=2
        )
    
    def test_classify_accuracy(self):
        """Test accuracy classification."""
        self.assertEqual(ValidationMetrics.classify_accuracy(1.0), "Excellent")
        self.assertEqual(ValidationMetrics.classify_accuracy(3.0), "Good")
        self.assertEqual(ValidationMetrics.classify_accuracy(7.0), "Acceptable")
        self.assertEqual(ValidationMetrics.classify_accuracy(15.0), "Poor")
    
    def test_calculate_correlation_difference(self):
        """Test correlation difference calculation."""
        diff, status = ValidationMetrics.calculate_correlation_difference(0.8, 0.77)
        self.assertAlmostEqual(diff, 0.03, places=3)
        self.assertEqual(status, "Excellent")
        
        diff, status = ValidationMetrics.calculate_correlation_difference(0.8, 0.66)
        self.assertAlmostEqual(diff, 0.14, places=3)
        self.assertEqual(status, "Acceptable")


class TestValidationCase(unittest.TestCase):
    """Test cases for ValidationCase dataclass."""
    
    def test_validation_case_creation(self):
        """Test ValidationCase object creation."""
        distributions = {
            'param1': {'type': 'normal', 'mean': 10, 'std': 2, 'enabled': True}
        }
        reference_data = {
            'output1': {'mean': 5.0, 'std': 1.0}
        }
        
        case = ValidationCase(
            name="Test Case",
            description="Test description",
            iterations=1000,
            seed=42,
            distributions=distributions,
            reference_data=reference_data
        )
        
        self.assertEqual(case.name, "Test Case")
        self.assertEqual(case.iterations, 1000)
        self.assertEqual(case.seed, 42)
        self.assertIsNone(case.sensitivity_data)


class TestATESValidationFramework(unittest.TestCase):
    """Test cases for ATESValidationFramework class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.framework = ATESValidationFramework(iterations=100, seed=42)
        
        # Mock data for testing
        self.mock_distributions = {
            'aquifer_temp': {
                'type': 'triangular', 
                'min': 10.0, 
                'max': 17.0, 
                'most_likely': 13.5, 
                'enabled': True
            }
        }
        
        self.mock_reference_data = {
            'heating_system_cop': {
                'mean': 3.45, 'std': 0.27, 'p10': 3.15, 'p50': 3.51, 'p90': 3.70
            }
        }
        
        self.mock_sensitivity_data = {
            'heating_system_cop': {
                'aquifer_temp': 0.99
            }
        }
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        self.assertEqual(self.framework.iterations, 100)
        self.assertEqual(self.framework.seed, 42)
        self.assertIsNone(self.framework.python_results)
        self.assertIsNone(self.framework.python_stats)
        self.assertEqual(len(self.framework.key_outputs), 20)
    
    @patch('validation.validation_framework.ATESMonteCarloEngine')
    @patch('validation.validation_framework.ATESParameters')
    def test_run_monte_carlo_simulation(self, mock_params, mock_engine):
        """Test Monte Carlo simulation execution."""
        # Mock successful simulation
        mock_results = pd.DataFrame({
            'success': [True] * 100,
            'heating_system_cop': np.random.normal(3.5, 0.3, 100)
        })
        
        mock_engine_instance = Mock()
        mock_engine_instance.run_simulation.return_value = mock_results
        mock_engine_instance.calculate_sensitivity_analysis.return_value = {}
        mock_engine.return_value = mock_engine_instance
        
        success = self.framework.run_monte_carlo_simulation(self.mock_distributions)
        
        self.assertTrue(success)
        self.assertIsNotNone(self.framework.python_results)
        mock_engine_instance.run_simulation.assert_called_once()
    
    def test_process_infinite_parameter(self):
        """Test processing of parameters with infinite values."""
        # Create test data with finite and infinite values
        data = pd.Series([1.0, 2.0, float('inf'), 3.0, np.nan])
        
        self.framework._process_infinite_parameter('test_param', data)
        
        self.assertIsNotNone(self.framework.python_stats)
        stats = self.framework.python_stats['test_param'] # type: ignore
        self.assertEqual(stats['finite_count'], 3)
        self.assertEqual(stats['infinite_count'], 1)
        self.assertEqual(stats['nan_count'], 1)
        self.assertAlmostEqual(stats['mean'], 2.0, places=2)
    
    def test_process_nan_sensitive_parameter(self):
        """Test processing of NaN-sensitive parameters."""
        data = pd.Series([1.0, 2.0, np.nan, 3.0, float('inf')])
        
        self.framework._process_nan_sensitive_parameter('test_param', data)
        
        self.assertIsNotNone(self.framework.python_stats)
        stats = self.framework.python_stats['test_param'] # type: ignore
        self.assertEqual(stats['finite_count'], 3)
        self.assertEqual(stats['nan_infinite_count'], 2)
        self.assertEqual(stats['finite_percentage'], 60.0)
    
    def test_process_standard_parameter(self):
        """Test processing of standard parameters."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        self.framework._process_standard_parameter('test_param', data)
        
        self.assertIsNotNone(self.framework.python_stats)
        stats = self.framework.python_stats['test_param'] # type: ignore
        self.assertEqual(stats['finite_count'], 5)
        self.assertEqual(stats['nan_count'], 0)
        self.assertEqual(stats['infinite_count'], 0)
        self.assertAlmostEqual(stats['mean'], 3.0, places=2)
        self.assertAlmostEqual(stats['p50'], 3.0, places=2)
    
    def test_validate_percentiles(self):
        """Test percentile validation."""
        # Set up mock statistics
        self.framework.python_stats = {
            'heating_system_cop': {
                'mean': 3.40, 'std': 0.25, 'p10': 3.10, 'p50': 3.45, 'p90': 3.65
            }
        }
        
        results = self.framework.validate_percentiles(self.mock_reference_data, verbose=False)
        
        self.assertIn('comparison_results', results)
        self.assertIn('total_compared', results)
        comparison_results = results['comparison_results']
        
        self.assertEqual(len(comparison_results), 1)
        self.assertEqual(comparison_results[0]['parameter'], 'heating_system_cop')
        self.assertIsNotNone(comparison_results[0]['mean_diff'])
        self.assertEqual(len(comparison_results[0]['percentile_comparisons']), 3)
    
    def test_validate_percentiles_no_stats(self):
        """Test percentile validation with no statistics."""
        # Don't set python_stats (leave as None)
        results = self.framework.validate_percentiles(self.mock_reference_data, verbose=False)
        
        self.assertIn('comparison_results', results)
        self.assertEqual(len(results['comparison_results']), 0)
        self.assertEqual(results['total_compared'], 0)
    
    def test_validate_sensitivity_analysis(self):
        """Test sensitivity analysis validation."""
        # Set up mock sensitivity results
        self.framework.sensitivity_results = {
            'heating_system_cop': pd.DataFrame({
                'Input_Parameter': ['aquifer_temp'],
                'Spearman_Correlation': [0.95],
                'Abs_Spearman': [0.95]
            })
        }
        
        results = self.framework.validate_sensitivity_analysis(
            self.mock_sensitivity_data, verbose=False
        )
        
        self.assertIn('validation_results', results)
        self.assertIn('summary', results)
        self.assertEqual(results['summary']['total'], 1)
    
    def test_validate_sensitivity_analysis_no_results(self):
        """Test sensitivity analysis validation with no results."""
        results = self.framework.validate_sensitivity_analysis(
            self.mock_sensitivity_data, verbose=False
        )
        
        self.assertIn('validation_results', results)
        self.assertIn('summary', results)
        self.assertEqual(results['summary']['total'], 0)
    
    def test_export_validation_report(self):
        """Test validation report export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up some mock validation results
            self.framework.validation_results = {
                'test_case': {
                    'case_name': 'test_case',
                    'statistical_validation': {'comparison_results': [], 'total_compared': 0},
                    'python_statistics': {}
                }
            }
            
            output_path = Path(temp_dir) / "test_report.json"
            result_path = self.framework.export_validation_report(str(output_path))
            
            self.assertTrue(result_path.exists())
            self.assertEqual(result_path, output_path)


class TestPredefinedValidationCases(unittest.TestCase):
    """Test cases for predefined validation scenarios."""
    
    def test_create_predefined_validation_cases(self):
        """Test creation of predefined validation cases."""
        cases = create_predefined_validation_cases()
        
        self.assertEqual(len(cases), 5)
        
        # Check case names
        case_names = [case.name for case in cases]
        expected_names = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]
        self.assertEqual(case_names, expected_names)
        
        # Check case 1 (quick test)
        case1 = cases[0]
        self.assertEqual(case1.iterations, 1000)
        self.assertEqual(case1.seed, 123)
        
        # Check case 5 (comprehensive)
        case5 = cases[4]
        self.assertEqual(case5.iterations, 10000)
        self.assertEqual(len(case5.distributions), 19)  # All parameters
    
    def test_validation_case_data_integrity(self):
        """Test that validation case data is properly structured."""
        cases = create_predefined_validation_cases()
        
        for case in cases:
            # Check required fields
            self.assertIsInstance(case.name, str)
            self.assertIsInstance(case.description, str)
            self.assertIsInstance(case.iterations, int)
            self.assertIsInstance(case.seed, int)
            self.assertIsInstance(case.distributions, dict)
            self.assertIsInstance(case.reference_data, dict)
            
            # Check distributions structure
            for param_name, dist_config in case.distributions.items():
                self.assertIn('type', dist_config)
                self.assertIn('enabled', dist_config)
                self.assertTrue(dist_config['enabled'])
            
            # Check reference data structure
            for output_name, ref_stats in case.reference_data.items():
                self.assertIn('mean', ref_stats)
                self.assertIsInstance(ref_stats['mean'], (int, float))


class TestIntegration(unittest.TestCase):
    """Integration tests for the validation framework."""
    
    def setUp(self):
        """Set up for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after integration tests."""
        shutil.rmtree(self.temp_dir)
    
    @patch('validation.validation_framework.ATESMonteCarloEngine')
    @patch('validation.validation_framework.ATESParameters')
    def test_run_validation_case_integration(self, mock_params, mock_engine):
        """Test complete validation case execution."""
        # Mock Monte Carlo results
        mock_results = pd.DataFrame({
            'success': [True] * 100,
            'heating_system_cop': np.random.normal(3.5, 0.3, 100),
            'heating_ave_power_to_building_MW': np.random.normal(10.0, 2.0, 100)
        })
        
        mock_sensitivity = {
            'heating_system_cop': pd.DataFrame({
                'Input_Parameter': ['aquifer_temp'],
                'Spearman_Correlation': [0.95],
                'Abs_Spearman': [0.95]
            })
        }
        
        mock_engine_instance = Mock()
        mock_engine_instance.run_simulation.return_value = mock_results
        mock_engine_instance.calculate_sensitivity_analysis.return_value = mock_sensitivity
        mock_engine.return_value = mock_engine_instance
        
        # Create test case
        test_case = ValidationCase(
            name="Integration Test",
            description="Integration test case",
            iterations=100,
            seed=42,
            distributions={
                'aquifer_temp': {
                    'type': 'triangular', 
                    'min': 10.0, 
                    'max': 17.0, 
                    'most_likely': 13.5, 
                    'enabled': True
                }
            },
            reference_data={
                'heating_system_cop': {
                    'mean': 3.45, 'std': 0.27, 'p10': 3.15, 'p50': 3.51, 'p90': 3.70
                }
            },
            sensitivity_data={
                'heating_system_cop': {
                    'aquifer_temp': 0.99
                }
            }
        )
        
        # Run validation
        framework = ATESValidationFramework()
        results = framework.run_validation_case(test_case, verbose=False)
        
        # Check results structure
        self.assertIn('case_name', results)
        self.assertIn('statistical_validation', results)
        self.assertIn('sensitivity_validation', results)
        self.assertIn('python_statistics', results)
        self.assertEqual(results['case_name'], "Integration Test")
    
    @patch('validation.validation_framework.run_comprehensive_validation_suite')
    def test_main_execution(self, mock_suite):
        """Test main execution path."""
        mock_suite.return_value = {}
        
        # Import and run main
        from validation import validation_framework
        
        # Verify the module can be imported and main functions exist
        self.assertTrue(hasattr(validation_framework, 'ATESValidationFramework'))
        self.assertTrue(hasattr(validation_framework, 'run_comprehensive_validation_suite'))
        self.assertTrue(hasattr(validation_framework, 'create_predefined_validation_cases'))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        framework = ATESValidationFramework()
        
        # Test empty series
        empty_data = pd.Series([], dtype=float)
        framework._process_standard_parameter('empty_param', empty_data)
        
        self.assertIsNotNone(framework.python_stats)
        stats = framework.python_stats['empty_param'] # type: ignore
        self.assertEqual(stats['finite_count'], 0)
        self.assertTrue(pd.isna(stats['mean']))
    
    def test_all_nan_data_handling(self):
        """Test handling of all-NaN data."""
        framework = ATESValidationFramework()
        
        # Test all NaN series
        nan_data = pd.Series([np.nan, np.nan, np.nan])
        framework._process_infinite_parameter('nan_param', nan_data)
        
        self.assertIsNotNone(framework.python_stats)
        stats = framework.python_stats['nan_param'] # type: ignore
        self.assertEqual(stats['finite_count'], 0)
        self.assertEqual(stats['nan_count'], 3)
    
    def test_invalid_reference_data(self):
        """Test handling of invalid reference data."""
        framework = ATESValidationFramework()
        framework.python_stats = {
            'valid_param': {'mean': 1.0, 'p10': 0.5, 'p50': 1.0, 'p90': 1.5}
        }
        
        # Missing parameter in reference data
        invalid_reference = {}
        
        results = framework.validate_percentiles(invalid_reference, verbose=False)
        self.assertIn('comparison_results', results)
        self.assertEqual(len(results['comparison_results']), 0)


if __name__ == '__main__':
    """
    Run the test suite.
    
    Usage:
        python test_validation_framework.py
    """
    
    # Configure test runner
    unittest.main(
        verbosity=2,
        buffer=True,
        warnings='ignore'
    )