"""
ATES Monte Carlo Validation Framework - Part 1

Validation tool for comparing Python ATES implementation with Crystal Ball results.
This module validates Python ATES Monte Carlo implementation against Crystal Ball reference data 
across multiple test cases with key validation metrics including power, energy, COP, and emissions.

Test Cases:
- Case 1: Quick validation test (1000 iterations)
- Case 2: Basic triangular distributions (seed=42)
- Case 3: Reproducibility test (seed=40)  
- Case 4: Complex mixed distributions
- Case 5: Comprehensive mixed distributions
"""

import pandas as pd
import numpy as np
import time
import warnings
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# Import ATES modules
from tool.core.ates_calculator import ATESParameters
from tool.core.monte_carlo_engine import ATESMonteCarloEngine, MonteCarloConfig, ParameterSampler, DistributionParams


@dataclass
class ValidationCase:
    """Data class to define a validation test case."""
    name: str
    description: str
    iterations: int
    seed: int
    distributions: Dict[str, Any]
    reference_data: Dict[str, Any]
    sensitivity_data: Optional[Dict[str, Any]] = None


class ValidationMetrics:
    """Class to calculate and store validation metrics."""
    
    @staticmethod
    def calculate_percentage_difference(observed: float, expected: float) -> float:
        """Calculate percentage difference between observed and expected values."""
        if expected == 0:
            return float('inf') if observed != 0 else 0.0
        return abs(observed - expected) / abs(expected) * 100
    
    @staticmethod
    def classify_accuracy(diff_percent: float) -> str:
        """Classify accuracy based on percentage difference."""
        if diff_percent <= 2:
            return "Excellent"
        elif diff_percent <= 5:
            return "Good"
        elif diff_percent <= 10:
            return "Acceptable"
        else:
            return "Poor"
    
    @staticmethod
    def calculate_correlation_difference(cb_corr: float, py_corr: float) -> Tuple[float, str]:
        """Calculate difference between correlation coefficients and classify."""
        difference = abs(cb_corr - py_corr)
        
        if difference <= 0.05:
            status = "Excellent"
        elif difference <= 0.10:
            status = "Good"
        elif difference <= 0.15:
            status = "Acceptable"
        else:
            status = "Poor"
        
        return difference, status
    
    @staticmethod
    def analyze_cop_distribution(data: pd.Series) -> Dict[str, Any]:
        total_count = len(data)
        
       
        finite_mask = np.isfinite(data)
        infinite_mask = ~finite_mask
        
        finite_data = data[finite_mask]
        finite_count = len(finite_data)
        infinite_count = infinite_mask.sum()
        
  
        direct_mode_rate = infinite_count / total_count * 100
        
        result = {
            'total_count': total_count,
            'direct_mode_count': infinite_count,
            'heat_pump_mode_count': finite_count,
            'direct_mode_rate_percent': direct_mode_rate,
            'is_cop_analysis': True  
        }
        
  
        if finite_count > 0:
            result.update({
                'mean': finite_data.mean(),
                'std': finite_data.std(),
                'p0': finite_data.min(),
                'p10': finite_data.quantile(0.10),
                'p20': finite_data.quantile(0.20),
                'p30': finite_data.quantile(0.30),
                'p40': finite_data.quantile(0.40),
                'p50': finite_data.quantile(0.50),
                'p60': finite_data.quantile(0.60),
                'p70': finite_data.quantile(0.70),
                'p80': finite_data.quantile(0.80),
                'p90': finite_data.quantile(0.90),
                'p100': finite_data.max(),
                'finite_count': finite_count
            })
        else:
            result.update({
                'mean': np.nan,
                'std': np.nan,
                'finite_count': 0,
                'note': 'All values are infinite (direct mode only)'
            })
        
        return result


class ATESValidationFramework:
    """
    Main validation framework for ATES Monte Carlo simulations.
    
    This class provides comprehensive validation capabilities including:
    - Statistical comparison with Crystal Ball reference data
    - Sensitivity analysis validation
    - Multi-case testing support
    - Detailed reporting and export functionality
    """
    
    def __init__(self, iterations: int = 10000, seed: int = 42):
        """
        Initialize validation framework.
        
        Args:
            iterations: Number of Monte Carlo iterations
            seed: Random seed for reproducibility
        """
        self.iterations = iterations
        self.seed = seed
        self.python_results = None
        self.python_stats = None
        self.sensitivity_results = None
        self.validation_results = {}
        
        # Key output parameters for validation
        self.key_outputs = [
            'heating_ave_power_to_HX_MW', 'cooling_ave_power_to_HX_MW',
            'heating_annual_energy_aquifer_GWhth', 'cooling_annual_energy_aquifer_GWhth',
            'energy_balance_ratio', 'volume_balance_ratio',
            'heating_ave_power_to_building_MW', 'cooling_ave_power_to_building_MW',
            'heating_annual_energy_building_GWhth', 'cooling_annual_energy_building_GWhth',
            'heating_monthly_to_building', 'cooling_monthly_to_building',
            'heating_annual_elec_energy_GWhe', 'cooling_annual_elec_energy_GWhe',
            'heating_system_cop', 'cooling_system_cop',
            'heating_elec_energy_per_thermal', 'cooling_elec_energy_per_thermal',
            'heating_co2_emissions_per_thermal', 'cooling_co2_emissions_per_thermal'
        ]
    
    def run_monte_carlo_simulation(self, distributions: Dict[str, Any]) -> bool:
        """
        Execute Monte Carlo simulation with specified distributions.
        
        Args:
            distributions: Dictionary defining parameter distributions
            
        Returns:
            bool: True if simulation completed successfully
        """
        print(f"Executing Monte Carlo simulation - Iterations: {self.iterations:,}, Seed: {self.seed}")
        
        base_params = ATESParameters()
        mc_config = MonteCarloConfig(
            iterations=self.iterations,
            seed=self.seed,
            parallel=True,
            max_workers=4
        )
        
        start_time = time.time()
        mc_engine = ATESMonteCarloEngine(base_params, mc_config)
        self.python_results = mc_engine.run_simulation(distributions)
        end_time = time.time()
        
        total = len(self.python_results)
        if self.python_results is not None:
            successful = len(self.python_results[self.python_results['success'] == True])
        else:
            successful = 0
        success_rate = successful / total * 100 if total > 0 else 0
        
        print(f"Completed in {end_time - start_time:.1f}s")
        print(f"Success rate: {success_rate:.1f}% ({successful:,}/{total:,})")
        
        self._calculate_statistics()
        self._extract_sensitivity_analysis(mc_engine, distributions)
        
        return True
    
    def _calculate_statistics(self):
        """Calculate detailed statistics with proper COP handling"""
        if self.python_results is None:
            return
        success_results = self.python_results[self.python_results['success'] == True]
        self.python_stats = {}
        
        # def COP params
        cop_parameters = [
            'heating_system_cop', 'cooling_system_cop',
            'heating_heat_pump_COP', 'cooling_heat_pump_COP'
        ]
        
        # Define parameter categories for special handling
        inf_params = [
            'heating_system_cop', 'cooling_system_cop', 
            'heating_heat_pump_COP', 'cooling_heat_pump_COP'
        ]
        
        nan_sensitive_params = [
            'heating_ave_power_to_building_MW', 'heating_ave_power_to_building_W',
            'heating_annual_energy_building_J', 'heating_annual_energy_building_kWhth', 
            'heating_annual_energy_building_GWhth', 'heating_monthly_to_building',
            'heating_annual_elec_energy_J', 'heating_annual_elec_energy_MWhe', 
            'heating_annual_elec_energy_GWhe', 'heating_elec_energy_HP',
            'cooling_ave_power_to_building_MW', 'cooling_ave_power_to_building_W',
            'cooling_annual_energy_building_J', 'cooling_annual_energy_building_kWhth',
            'cooling_annual_energy_building_GWhth', 'cooling_monthly_to_building', 
            'cooling_annual_elec_energy_J', 'cooling_annual_elec_energy_MWhe',
            'cooling_annual_elec_energy_GWhe', 'cooling_elec_energy_HP'
        ]
        
        for param in self.key_outputs:
            if param not in success_results.columns:
                continue
                
            original_data = success_results[param]
            
            if param in cop_parameters:
                self.python_stats[param] = ValidationMetrics.analyze_cop_distribution(original_data)
            elif param in inf_params:
                self._process_infinite_parameter(param, original_data)
            elif param in nan_sensitive_params:
                self._process_nan_sensitive_parameter(param, original_data)
            else:
                self._process_standard_parameter(param, original_data)
    
    def _process_infinite_parameter(self, param: str, data: pd.Series):
        """Process parameters that may legitimately contain infinite values (like COP)."""
        if self.python_stats is None:
            self.python_stats = {}
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            self.python_stats[param] = {
                'mean': np.nan, 'std': np.nan,
                'total_count': len(data), 'nan_count': len(data), 
                'finite_count': 0, 'infinite_count': 0
            }
            return
        
        finite_data = clean_data[np.isfinite(clean_data)]
        infinite_data = clean_data[~np.isfinite(clean_data)]
        
        nan_count = len(data) - len(clean_data)
        finite_count = len(finite_data)
        infinite_count = len(infinite_data)
        
        if finite_count > 0:
            self.python_stats[param] = {
                'mean': finite_data.mean(),
                'std': finite_data.std(),
                'p0': finite_data.min(),
                'p10': finite_data.quantile(0.10),
                'p20': finite_data.quantile(0.20),
                'p30': finite_data.quantile(0.30),
                'p40': finite_data.quantile(0.40),
                'p50': finite_data.quantile(0.50),
                'p60': finite_data.quantile(0.60),
                'p70': finite_data.quantile(0.70),
                'p80': finite_data.quantile(0.80),
                'p90': finite_data.quantile(0.90),
                'p100': finite_data.max(),
                'total_count': len(data),
                'nan_count': nan_count,
                'finite_count': finite_count,
                'infinite_count': infinite_count
            }
        else:
            self.python_stats[param] = {
                'mean': float('inf') if infinite_count > 0 else np.nan,
                'std': 0 if infinite_count > 0 else np.nan,
                'total_count': len(data),
                'nan_count': nan_count,
                'finite_count': 0,
                'infinite_count': infinite_count
            }
    
    def _process_nan_sensitive_parameter(self, param: str, data: pd.Series):
        """Process parameters that may contain NaN due to infinite ehp calculations."""
        if self.python_stats is None:
            self.python_stats = {}
        finite_data = data[np.isfinite(data)]
        
        nan_count = len(data) - len(finite_data)
        finite_count = len(finite_data)
        
        if finite_count > 0:
            self.python_stats[param] = {
                'mean': finite_data.mean(),
                'std': finite_data.std(),
                'p0': finite_data.min(),
                'p10': finite_data.quantile(0.10),
                'p20': finite_data.quantile(0.20),
                'p30': finite_data.quantile(0.30),
                'p40': finite_data.quantile(0.40),
                'p50': finite_data.quantile(0.50),
                'p60': finite_data.quantile(0.60),
                'p70': finite_data.quantile(0.70),
                'p80': finite_data.quantile(0.80),
                'p90': finite_data.quantile(0.90),
                'p100': finite_data.max(),
                'total_count': len(data),
                'nan_infinite_count': nan_count,
                'finite_count': finite_count,
                'finite_percentage': (finite_count / len(data)) * 100 if len(data) > 0 else 0
            }
        else:
            self.python_stats[param] = {
                'mean': np.nan,
                'std': np.nan,
                'total_count': len(data),
                'nan_infinite_count': nan_count,
                'finite_count': 0,
                'finite_percentage': 0
            }
    
    def _process_standard_parameter(self, param: str, data: pd.Series):
        """Process standard parameters by removing NaN and infinite values."""
        if self.python_stats is None:
            self.python_stats = {}
        clean_data = data.dropna()
        finite_data = clean_data[np.isfinite(clean_data)]
        
        nan_count = len(data) - len(clean_data)
        infinite_count = len(clean_data) - len(finite_data)
        finite_count = len(finite_data)
        
        if finite_count > 0:
            self.python_stats[param] = {
                'mean': finite_data.mean(),
                'std': finite_data.std(),
                'p0': finite_data.min(),
                'p10': finite_data.quantile(0.10),
                'p20': finite_data.quantile(0.20),
                'p30': finite_data.quantile(0.30),
                'p40': finite_data.quantile(0.40),
                'p50': finite_data.quantile(0.50),
                'p60': finite_data.quantile(0.60),
                'p70': finite_data.quantile(0.70),
                'p80': finite_data.quantile(0.80),
                'p90': finite_data.quantile(0.90),
                'p100': finite_data.max(),
                'total_count': len(data),
                'nan_count': nan_count,
                'infinite_count': infinite_count,
                'finite_count': finite_count
            }
        else:
            self.python_stats[param] = {
                'mean': np.nan,
                'std': np.nan,
                'total_count': len(data),
                'nan_count': nan_count,
                'infinite_count': infinite_count,
                'finite_count': 0
            }
    
    def _extract_sensitivity_analysis(self, mc_engine, distributions):
        """Extract sensitivity analysis results from Monte Carlo engine."""
        rng = np.random.default_rng(self.seed)
        enabled_params = {name: config for name, config in distributions.items() 
                         if config['type'] != 'single_value'}
        
        samples = {}
        for param_name, dist_config in enabled_params.items():
            dist_params = DistributionParams(
                type=dist_config['type'],
                value=float(dist_config.get('value', 0.0)),
                min_val=float(dist_config.get('min', 0.0)),
                max_val=float(dist_config.get('max', 0.0)),
                most_likely=float(dist_config.get('most_likely', 0.0)),
                mean=float(dist_config.get('mean', 0.0)),
                std=float(dist_config.get('std', 0.0))
            )
            samples[param_name] = ParameterSampler.sample_parameter(
                dist_params, self.iterations, rng
            )
        
        parameter_samples_df = pd.DataFrame(samples)
        if self.python_results is not None:
            mc_engine.results = self.python_results
            self.sensitivity_results = mc_engine.calculate_sensitivity_analysis(parameter_samples_df)
        else:
            self.sensitivity_results = {}
    
    def _compare_cop_parameter(self, param: str, python_stats: Dict, cb_stats: Dict, verbose: bool):
        direct_rate = python_stats.get('direct_mode_rate_percent', 0)
        
        if verbose:
            print(f"{param:<35} {'Info':<8} {'Direct:':<12} {direct_rate:<12.1f}% {'':<8} {'COP Analysis'}")
        
        if python_stats.get('heat_pump_mode_count', 0) > 0:
            py_mean = python_stats.get('mean', 0.0)
            cb_mean = cb_stats.get('mean', 0.0)
            
            if py_mean != 0 and py_mean != float('inf'):
                mean_diff = ValidationMetrics.calculate_percentage_difference(py_mean, cb_mean)
                mean_status = ValidationMetrics.classify_accuracy(mean_diff)
                
                if verbose:
                    print(f"{'':<35} {'Mean':<8} {py_mean:<12.4f} {cb_mean:<12.4f} {mean_diff:<8.1f} {mean_status}")
                    
                    
                    for pct in ['p10', 'p50', 'p90']:
                        py_val = python_stats.get(pct, 0.0)
                        cb_val = cb_stats.get(pct, 0.0)
                        
                        if py_val != 0:
                            pct_diff = ValidationMetrics.calculate_percentage_difference(py_val, cb_val)
                            pct_status = ValidationMetrics.classify_accuracy(pct_diff)
                            print(f"{'':<35} {pct.upper():<8} {py_val:<12.4f} {cb_val:<12.4f} {pct_diff:<8.1f} {pct_status}")
        else:
            if verbose:
                print(f"{'':<35} {'Note':<8} {'All values':<12} {'are infinite':<12} {'':<8} {'No comparison'}")
        
        if verbose:
            print()


    def validate_percentiles(self, crystal_ball_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Compare Python results with Crystal Ball reference data.
        
        Args:
            crystal_ball_data: Reference data from Crystal Ball
            verbose: Whether to print detailed comparison results
            
        Returns:
            Dictionary containing comparison results
        """
        if verbose:
            print("STATISTICAL COMPARISON RESULTS")
            print("=" * 80)
            print(f"{'Parameter':<35} {'Metric':<8} {'Python':<12} {'Crystal Ball':<12} {'Diff %':<8} {'Status'}")
            print("-" * 80)
        
        comparison_results = []
        
        if self.python_stats is None:
            if verbose:
                print("No Python statistics available")
            return {
                'comparison_results': comparison_results,
                'total_compared': 0,
                'summary': {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
            }
        
        for param in self.key_outputs:
            if param in self.python_stats and param in crystal_ball_data:
                python_param = self.python_stats[param]
                cb_param = crystal_ball_data[param]
                
            
                if python_param.get('is_cop_analysis', False):
                    self._compare_cop_parameter(param, python_param, cb_param, verbose)
                else:
                  
                    py_mean = python_param.get('mean', 0.0)
                    cb_mean = cb_param.get('mean', 0.0)
                    
                    mean_diff = None
                    mean_status = None
                    
                    if py_mean != 0 and py_mean != float('inf'):
                        mean_diff = ValidationMetrics.calculate_percentage_difference(py_mean, cb_mean)
                        mean_status = ValidationMetrics.classify_accuracy(mean_diff)
                        
                        if verbose:
                            print(f"{param:<35} {'Mean':<8} {py_mean:<12.4f} {cb_mean:<12.4f} {mean_diff:<8.1f} {mean_status}")
                    
                    
                    percentile_comparisons = []
                    for pct in ['p10', 'p50', 'p90']:
                        py_val = python_param.get(pct, 0.0)
                        cb_val = cb_param.get(pct, 0.0)
                        
                        if py_val != 0:
                            pct_diff = ValidationMetrics.calculate_percentage_difference(py_val, cb_val)
                            pct_status = ValidationMetrics.classify_accuracy(pct_diff)
                            
                            if verbose:
                                print(f"{'':<35} {pct.upper():<8} {py_val:<12.4f} {cb_val:<12.4f} {pct_diff:<8.1f} {pct_status}")
                            
                            percentile_comparisons.append({
                                'percentile': pct,
                                'python_value': py_val,
                                'crystal_ball_value': cb_val,
                                'difference_percent': pct_diff,
                                'status': pct_status
                            })
                    
                    comparison_results.append({
                        'parameter': param,
                        'mean_diff': mean_diff,
                        'mean_status': mean_status,
                        'percentile_comparisons': percentile_comparisons
                    })
                    
                    if verbose:
                        print()
        
        if verbose:
            self._print_validation_summary(comparison_results)
        
        return {
            'comparison_results': comparison_results,
            'total_compared': len(comparison_results),
            'summary': self._calculate_comparison_summary(comparison_results)
        }

    def _calculate_comparison_summary(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate summary statistics for comparison results."""
        summary = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
        
        for result in comparison_results:
            if result.get('mean_status'):
                status = result['mean_status'].lower()
                if status in summary:
                    summary[status] += 1
        
        return summary
    
    def validate_sensitivity_analysis(self, cb_rank_correlation_data: Dict[str, Dict[str, float]], 
                                    verbose: bool = True) -> Dict[str, Any]:
        """
        Compare Python Spearman correlation results with Crystal Ball rank correlation data.
        
        Args:
            cb_rank_correlation_data: Crystal Ball rank correlation reference data
            verbose: Whether to print detailed comparison results
            
        Returns:
            Dictionary containing sensitivity validation results
        """
        validation_results = []
        
        if verbose:
            print("SENSITIVITY ANALYSIS COMPARISON")
            print("=" * 80)
            print(f"{'Output Parameter':<35} {'Input Parameter':<25} {'CB Rank':<10} {'Python':<10} {'Diff':<8} {'Status'}")
            print("-" * 80)

        if self.sensitivity_results is None:
            if verbose:
                print("No sensitivity analysis results available")
            return {
                'validation_results': validation_results,
                'summary': {'total': 0, 'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'success_rate': 0, 'validation_passed': False}
            }
        
        for output_param, cb_data in cb_rank_correlation_data.items():
            if output_param not in self.sensitivity_results:
                continue
                
            python_sensitivity_df = self.sensitivity_results[output_param]
            param_comparisons = []
            
            for cb_input_param, cb_rank_correlation in cb_data.items():
                python_row = python_sensitivity_df[
                    python_sensitivity_df['Input_Parameter'] == cb_input_param
                ]
                
                if len(python_row) == 0:
                    continue
                
                python_spearman_corr = python_row.iloc[0]['Spearman_Correlation']
                difference, status = ValidationMetrics.calculate_correlation_difference(
                    cb_rank_correlation, python_spearman_corr
                )
                
                if verbose:
                    print(f"{output_param if cb_input_param == list(cb_data.keys())[0] else '':<35} "
                          f"{cb_input_param:<25} {cb_rank_correlation:<10.3f} {python_spearman_corr:<10.3f} "
                          f"{difference:<8.3f} {status}")
                
                comparison = {
                    'output_parameter': output_param,
                    'input_parameter': cb_input_param,
                    'cb_rank_correlation': cb_rank_correlation,
                    'python_rank_correlation': python_spearman_corr,
                    'difference': difference,
                    'status': status
                }
                
                param_comparisons.append(comparison)
            
            validation_results.append({
                'output_parameter': output_param,
                'comparisons': param_comparisons
            })
            
            if verbose:
                print()
        
        # Calculate summary
        all_comparisons = []
        for result in validation_results:
            all_comparisons.extend(result['comparisons'])
        
        summary = self._calculate_sensitivity_summary(all_comparisons)
        
        if verbose:
            self._print_sensitivity_summary(summary)
        
        return {
            'validation_results': validation_results,
            'summary': summary
        }

    def run_validation_case(self, case: ValidationCase, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a complete validation case.
        
        Args:
            case: ValidationCase object defining the test
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary containing all validation results
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"VALIDATION CASE: {case.name}")
            print(f"Description: {case.description}")
            print(f"{'='*80}")
        
        # Update framework parameters for this case
        self.iterations = case.iterations
        self.seed = case.seed
        
        # Run simulation
        success = self.run_monte_carlo_simulation(case.distributions)
        if not success:
            return {"error": "Simulation failed"}
        
        # Validate statistics
        stats_validation = self.validate_percentiles(case.reference_data, verbose)
        
        # Validate sensitivity if data available
        sensitivity_validation = None
        if case.sensitivity_data:
            sensitivity_validation = self.validate_sensitivity_analysis(case.sensitivity_data, verbose)
        
        results = {
            'case_name': case.name,
            'case_description': case.description,
            'simulation_params': {
                'iterations': case.iterations,
                'seed': case.seed
            },
            'statistical_validation': stats_validation,
            'sensitivity_validation': sensitivity_validation,
            'python_statistics': self.python_stats,
            'raw_results': self.python_results
        }
        
        self.validation_results[case.name] = results
        return results

    def export_validation_report(self, output_path: str = "validation_report.json"):
        """
        Export comprehensive validation report to JSON file.
        
        Args:
            output_path: Path for output file
        """
        report = {
            'validation_framework_version': '1.0.0',
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'total_cases_run': len(self.validation_results),
            'validation_cases': {}
        }
        
        for case_name, case_results in self.validation_results.items():
            # Clean results for JSON serialization
            cleaned_results = self._clean_results_for_export(case_results)
            report['validation_cases'][case_name] = cleaned_results
        
        # Write to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Validation report exported to: {output_file.absolute()}")
        return output_file

    def _clean_results_for_export(self, results: Dict) -> Dict:
        """Clean results dictionary for JSON serialization."""
        cleaned = {}
        for key, value in results.items():
            if key == 'raw_results':
                continue
            elif isinstance(value, pd.DataFrame):
                cleaned[key] = value.to_dict('records')
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                cleaned[key] = float(value)
            elif pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned

    def _calculate_sensitivity_summary(self, all_comparisons: List[Dict]) -> Dict:
        """Calculate summary statistics for sensitivity validation."""
        total = len(all_comparisons)
        excellent = sum(1 for c in all_comparisons if c['status'] == 'Excellent')
        good = sum(1 for c in all_comparisons if c['status'] == 'Good')
        acceptable = sum(1 for c in all_comparisons if c['status'] == 'Acceptable')
        poor = sum(1 for c in all_comparisons if c['status'] == 'Poor')
        
        success_rate = (excellent + good) / total * 100 if total > 0 else 0
        
        return {
            'total': total,
            'excellent': excellent,
            'good': good,
            'acceptable': acceptable,
            'poor': poor,
            'success_rate': success_rate,
            'validation_passed': success_rate >= 60
        }
    
    def _print_validation_summary(self, comparison_results: List[Dict]):
        """Print comprehensive validation summary."""
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        mean_diffs = [r['mean_diff'] for r in comparison_results 
                     if r['mean_diff'] is not None]
        
        if mean_diffs:
            excellent = sum(1 for d in mean_diffs if d <= 2)
            good = sum(1 for d in mean_diffs if 2 < d <= 5)
            acceptable = sum(1 for d in mean_diffs if 5 < d <= 10)
            poor = sum(1 for d in mean_diffs if d > 10)
            
            print(f"Mean comparison results:")
            print(f"  Excellent (â‰¤2%): {excellent}/{len(mean_diffs)}")
            print(f"  Good (2-5%): {good}/{len(mean_diffs)}")
            print(f"  Acceptable (5-10%): {acceptable}/{len(mean_diffs)}")
            print(f"  Poor (>10%): {poor}/{len(mean_diffs)}")
            
            if poor == 0 and acceptable <= 1:
                print("VALIDATION PASSED: High consistency between tools")
            elif poor == 0:
                print("VALIDATION ACCEPTABLE: Minor differences detected")
            else:
                print("VALIDATION REQUIRES INVESTIGATION: Significant differences found")
    
    def _print_sensitivity_summary(self, summary: Dict):
        """Print sensitivity analysis validation summary."""
        print("=" * 80)
        print("SENSITIVITY VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Excellent (â‰¤0.05): {summary['excellent']}/{summary['total']}")
        print(f"Good (â‰¤0.10): {summary['good']}/{summary['total']}")
        print(f"Acceptable (â‰¤0.15): {summary['acceptable']}/{summary['total']}")
        print(f"Poor (>0.15): {summary['poor']}/{summary['total']}")
        print(f"Overall success rate: {summary['success_rate']:.1f}%")
        print(f"Validation status: {'PASSED' if summary['validation_passed'] else 'REQUIRES INVESTIGATION'}")


def create_predefined_validation_cases() -> List[ValidationCase]:
    """
    Create predefined validation cases based on Crystal Ball reference data.
    
    Returns:
        List of ValidationCase objects ready for testing
    """
    cases = []
    
    # Basic triangular distributions for Cases 1, 2 & 3
    basic_triangular_distributions = {
        'aquifer_temp': {
            'type': 'triangular', 'min': 10.0, 'max': 17.0, 'most_likely': 13.5
        },
        'thermal_recovery_factor': {
            'type': 'triangular', 'min': 0.25, 'max': 0.65, 'most_likely': 0.4
        },
        'heating_months': {
            'type': 'triangular', 'min': 5.0, 'max': 7.5, 'most_likely': 6.5
        },
        'cooling_months': {
            'type': 'triangular', 'min': 2.5, 'max': 4.5, 'most_likely': 3.5
        }
    }
    
    # Crystal Ball reference data for basic triangular distributions
    triangular_reference_data = {
        'heating_ave_power_to_HX_MW': {
            'mean': 8.01, 'std': 3.32, 'p10': 3.56, 'p50': 7.99, 'p90': 12.49
        },
        'cooling_ave_power_to_HX_MW': {
            'mean': 15.54, 'std': 6.89, 'p10': 6.72, 'p50': 15.12, 'p90': 24.67
        },
        'heating_annual_energy_aquifer_GWhth': {
            'mean': 37.78, 'std': 16.05, 'p10': 16.59, 'p50': 37.41, 'p90': 59.31
        },
        'cooling_annual_energy_aquifer_GWhth': {
            'mean': 39.92, 'std': 16.89, 'p10': 17.61, 'p50': 39.56, 'p90': 62.71
        },
        'energy_balance_ratio': {
            'mean': 0.0279, 'std': 0.0055, 'p10': 0.0203, 'p50': 0.0280, 'p90': 0.0352
        },
        'volume_balance_ratio': {
            'mean': 0.31, 'std': 0.271, 'p10': -0.051, 'p50': 0.305, 'p90': 0.682
        },
        'heating_ave_power_to_building_MW': {
            'mean': 10.86, 'std': 4.43, 'p10': 4.89, 'p50': 10.85, 'p90': 16.80
        },
        'cooling_ave_power_to_building_MW': {
            'mean': 15.54, 'std': 6.89, 'p10': 6.72, 'p50': 15.12, 'p90': 24.68
        },
        'heating_annual_energy_building_GWhth': {
            'mean': 51.22, 'std': 21.44, 'p10': 22.83, 'p50': 50.87, 'p90': 79.83
        },
        'cooling_annual_energy_building_GWhth': {
            'mean': 39.92, 'std': 16.89, 'p10': 17.61, 'p50': 39.56, 'p90': 62.71
        },
        'heating_monthly_to_building': {
            'mean': 8.08, 'std': 3.29, 'p10': 3.64, 'p50': 8.07, 'p90': 12.50
        },
        'cooling_monthly_to_building': {
            'mean': 11.56, 'std': 5.12, 'p10': 5.0, 'p50': 11.25, 'p90': 18.36
        },
        'heating_annual_elec_energy_GWhe': {
            'mean': 14.48, 'std': 5.42, 'p10': 7.22, 'p50': 14.46, 'p90': 21.68
        },
        'cooling_annual_elec_energy_GWhe': {
            'mean': 0.62, 'std': 0.38, 'p10': 0.20, 'p50': 0.55, 'p90': 1.16
        },
        'heating_system_cop': {
            'mean': 3.45, 'std': 0.27, 'p10': 3.15, 'p50': 3.51, 'p90': 3.70
        },
        'cooling_system_cop': {
            'mean': 72.17, 'std': 14.25, 'p10': 53.25, 'p50': 72.03, 'p90': 91.21
        },
        'heating_elec_energy_per_thermal': {
            'mean': 0.292, 'std': 0.034, 'p10': 0.270, 'p50': 0.285, 'p90': 0.317
        },
        'cooling_elec_energy_per_thermal': {
            'mean': 0.0145, 'std': 0.0031, 'p10': 0.0110, 'p50': 0.0139, 'p90': 0.0188
        },
        'heating_co2_emissions_per_thermal': {
            'mean': 52.57, 'std': 6.09, 'p10': 48.62, 'p50': 51.27, 'p90': 57.11
        },
        'cooling_co2_emissions_per_thermal': {
            'mean': 2.6, 'std': 0.56, 'p10': 1.97, 'p50': 2.50, 'p90': 3.38
        }
    }
    
    # Crystal Ball rank correlation data for Cases 1, 2, 3 sensitivity validation
    basic_sensitivity_data = {
        'cooling_ave_power_to_building_MW': {
            'aquifer_temp': 0.94,
            'cooling_months': -0.24,
            'heating_months': 0.18,
            'thermal_recovery_factor': 0.05
        },
        'cooling_system_cop': {
            'aquifer_temp': -0.99,
            'thermal_recovery_factor': 0.12,
            'heating_months': -0.01,
            'cooling_months': -0.01
        },
        'energy_balance_ratio': {
            'thermal_recovery_factor': -1.00,
            'heating_months': -0.02,
            'cooling_months': 0.00,
            'aquifer_temp': 0.00
        },
        'heating_ave_power_to_building_MW': {
            'aquifer_temp': 1.00,
            'thermal_recovery_factor': 0.06,
            'heating_months': 0.01,
            'cooling_months': 0.01
        },
        'heating_system_cop': {
            'aquifer_temp': 0.99,
            'thermal_recovery_factor': 0.13,
            'heating_months': 0.01,
            'cooling_months': 0.01
        },
        'volume_balance_ratio': {
            'aquifer_temp': -1.00,
            'heating_months': -0.01,
            'cooling_months': -0.01,
            'thermal_recovery_factor': 0.00
        },
        'heating_annual_energy_building_GWhth': {
            'aquifer_temp': 0.98,
            'heating_months': 0.20,
            'thermal_recovery_factor': 0.06,
            'cooling_months': 0.01
        },
        'cooling_annual_energy_building_GWhth': {
            'aquifer_temp': 0.98,
            'heating_months': 0.19,
            'thermal_recovery_factor': 0.05,
            'cooling_months': 0.01
        }
    }
    
    # Case 1: Quick validation test
    cases.append(ValidationCase(
        name="Case 1",
        description="Quick validation test (1000 iterations)",
        iterations=1000,
        seed=123,
        distributions=basic_triangular_distributions,
        reference_data=triangular_reference_data,
        sensitivity_data=basic_sensitivity_data
    ))
    
    # Case 2: Basic triangular distributions
    cases.append(ValidationCase(
        name="Case 2",
        description="Basic triangular distributions validation (seed=42)",
        iterations=10000,
        seed=42,
        distributions=basic_triangular_distributions,
        reference_data=triangular_reference_data,
        sensitivity_data=basic_sensitivity_data
    ))
    
    # Case 3: Reproducibility test
    cases.append(ValidationCase(
        name="Case 3",
        description="Reproducibility test (seed=40)",
        iterations=10000,
        seed=40,
        distributions=basic_triangular_distributions,
        reference_data=triangular_reference_data,
        sensitivity_data=basic_sensitivity_data
    ))
    # Case 4: Complex mixed distributions
    complex_mixed_distributions = {
        'aquifer_temp': {'type': 'normal', 'mean': 13.5, 'std': 1.2},
        'thermal_recovery_factor': {'type': 'triangular', 'min': 0.3, 'max': 0.6, 'most_likely': 0.4},
        'heating_months': {'type': 'range', 'min': 5.8, 'max': 7.2},
        'pump_energy_density': {'type': 'lognormal', 'mean': 600, 'std': 50}
    }

    # Case 5: Comprehensive mixed distributions (all 19 parameters)
    comprehensive_mixed_distributions = {
        'aquifer_temp': {'type': 'normal', 'mean': 13.5, 'std': 2.0},
        'water_density': {'type': 'triangular', 'min': 995.0, 'max': 1005.0, 'most_likely': 1000.0},
        'water_specific_heat_capacity': {'type': 'range', 'min': 4150.0, 'max': 4220.0},
        'thermal_recovery_factor': {'type': 'lognormal', 'mean': 0.4, 'std': 0.08},
        'heating_target_avg_flowrate_pd': {'type': 'normal', 'mean': 60.0, 'std': 15.0},
        'tolerance_in_energy_balance': {'type': 'triangular', 'min': 0.05, 'max': 0.25, 'most_likely': 0.15},
        'heating_number_of_doublets': {'type': 'range', 'min': 18, 'max': 26},
        'heating_months': {'type': 'lognormal', 'mean': 6.5, 'std': 0.8},
        'cooling_months': {'type': 'normal', 'mean': 3.5, 'std': 0.6},
        'pump_energy_density': {'type': 'triangular', 'min': 400.0, 'max': 800.0, 'most_likely': 600.0},
        'heating_ave_injection_temp': {'type': 'range', 'min': 8.0, 'max': 12.0,},
        'heating_temp_to_building': {'type': 'triangular', 'min': 45.0, 'max': 75.0, 'most_likely': 60.0},
        'cop_param_a': {'type': 'normal', 'mean': 100.0, 'std': 20.0},
        'cop_param_b': {'type': 'triangular', 'min': 1.2, 'max': 2.0, 'most_likely': 1.6},
        'cop_param_c': {'type': 'range', 'min': -0.12, 'max': -0.04},
        'cop_param_d': {'type': 'lognormal', 'mean': 7.0, 'std': 1.2},
        'carbon_intensity': {'type': 'normal', 'mean': 180.0, 'std': 40.0},
        'cooling_ave_injection_temp': {'type': 'triangular', 'min': 19.0, 'max': 23.0, 'most_likely': 21.0},
        'cooling_temp_to_building': {'type': 'range', 'min': 12.0, 'max': 16.0}
    }
    
    # Case 4: Complex mixed distributions reference data 
    complex_mixed_reference_data = {
        'heating_ave_power_to_HX_MW': {
            'mean': 8.06, 'std': 2.74, 'p10': 4.57, 'p50': 8.04, 'p90': 11.61
        },
        'cooling_ave_power_to_HX_MW': {
            'mean': 15.81, 'std': 5.46, 'p10': 8.91, 'p50': 15.68, 'p90': 22.85
        },
        'heating_annual_energy_aquifer_GWhth': {
            'mean': 38.97, 'std': 13.50, 'p10': 22.00, 'p50': 38.60, 'p90': 56.39
        },
        'cooling_annual_energy_aquifer_GWhth': {
            'mean': 41.18, 'std': 14.21, 'p10': 23.21, 'p50': 40.84, 'p90': 59.50
        },
        'energy_balance_ratio': {
            'mean': 0.0278, 'std': 0.0042, 'p10': 0.0219, 'p50': 0.0281, 'p90': 0.0333
        },
        'volume_balance_ratio': {
            'mean': 0.303, 'std': 0.225, 'p10': 0.017, 'p50': 0.299, 'p90': 0.595
        },
        'heating_ave_power_to_building_MW': {
            'mean': 10.94, 'std': 3.66, 'p10': 6.25, 'p50': 10.93, 'p90': 15.66
        },
        'cooling_ave_power_to_building_MW': {
            'mean': 15.82, 'std': 5.46, 'p10': 8.91, 'p50': 15.68, 'p90': 22.85
        },
        'heating_annual_energy_building_GWhth': {
            'mean': 52.88, 'std': 18.03, 'p10': 30.07, 'p50': 52.50, 'p90': 76.13
        },
        'cooling_annual_energy_building_GWhth': {
            'mean': 41.18, 'std': 14.21, 'p10': 23.21, 'p50': 40.84, 'p90': 59.50
        },
        'heating_monthly_to_building': {
            'mean': 8.14, 'std': 2.72, 'p10': 4.65, 'p50': 8.13, 'p90': 11.65
        },
        'cooling_monthly_to_building': {
            'mean': 11.77, 'std': 4.06, 'p10': 6.63, 'p50': 11.67, 'p90': 17.00
        },
        'heating_annual_elec_energy_GWhe': {
            'mean': 14.97, 'std': 4.55, 'p10': 9.16, 'p50': 14.94, 'p90': 20.77
        },
        'cooling_annual_elec_energy_GWhe': {
            'mean': 0.62, 'std': 0.33, 'p10': 0.27, 'p50': 0.57, 'p90': 1.03
        },
        'heating_system_cop': {
            'mean': 3.49, 'std': 0.96, 'p10': 3.27, 'p50': 3.51, 'p90': 3.67
        },
        'cooling_system_cop': {
            'mean': 72.50, 'std': 13.21, 'p10': 55.91, 'p50': 72.03, 'p90': 89.49
        },
        'heating_elec_energy_per_thermal': {
            'mean': 0.288, 'std': 0.045, 'p10': 0.272, 'p50': 0.285, 'p90': 0.306
        },
        'cooling_elec_energy_per_thermal': {
            'mean': 0.0143, 'std': 0.0028, 'p10': 0.0112, 'p50': 0.0139, 'p90': 0.0179
        },
        'heating_co2_emissions_per_thermal': {
            'mean': 51.91, 'std': 8.15, 'p10': 48.99, 'p50': 51.24, 'p90': 55.02
        },
        'cooling_co2_emissions_per_thermal': {
            'mean': 2.57, 'std': 0.51, 'p10': 2.01, 'p50': 2.50, 'p90': 3.22
        }
    }

    # Case 4 sensitivity data
    case4_sensitivity_data = {
        'volume_balance_ratio': {
            'aquifer_temp': -1.00,
            'pump_energy_density': -0.02,
            'heating_months': -0.01,
            'thermal_recovery_factor': 0.00
        },
        'heating_system_cop': {
            'aquifer_temp': 0.93,
            'thermal_recovery_factor': 0.31,
            'pump_energy_density': -0.12,
            'heating_months': 0.01
        },
        'heating_ave_power_to_building_MW': {
            'aquifer_temp': 0.99,
            'thermal_recovery_factor': 0.13,
            'pump_energy_density': 0.02,
            'heating_months': 0.01
        },
        'energy_balance_ratio': {
            'thermal_recovery_factor': -1.00,
            'heating_months': -0.01,
            'pump_energy_density': 0.00,
            'aquifer_temp': 0.00
        },
        'cooling_system_cop': {
            'aquifer_temp': -0.86,
            'pump_energy_density': -0.45,
            'thermal_recovery_factor': 0.21,
            'heating_months': -0.02
        },
        'cooling_ave_power_to_building_MW': {
            'aquifer_temp': 0.97,
            'heating_months': 0.19,
            'thermal_recovery_factor': 0.11,
            'pump_energy_density': 0.02
        },
        'heating_annual_energy_building_GWhth': {
            'aquifer_temp': 0.97,
            'heating_months': 0.19,
            'thermal_recovery_factor': 0.13,
            'pump_energy_density': 0.02
        },
        'cooling_annual_energy_building_GWhth': {
            'aquifer_temp': 0.97,
            'heating_months': 0.19,
            'thermal_recovery_factor': 0.11,
            'pump_energy_density': 0.02
        }
    }

    # Case 5: Comprehensive mixed distributions reference data 
    comprehensive_reference_data = {
        'heating_ave_power_to_HX_MW': {
            'mean': 7.92, 'std': 5.83, 'p10': 1.11, 'p50': 7.39, 'p90': 15.56
        },
        'cooling_ave_power_to_HX_MW': {
            'mean': 16.13, 'std': 12.75, 'p10': 2.17, 'p50': 14.55, 'p90': 32.48
        },
        'heating_annual_energy_aquifer_GWhth': {
            'mean': 38.38, 'std': 29.03, 'p10': 5.30, 'p50': 35.23, 'p90': 76.00
        },
        'cooling_annual_energy_aquifer_GWhth': {
            'mean': 40.75, 'std': 30.80, 'p10': 5.63, 'p50': 37.39, 'p90': 80.76
        },
        'energy_balance_ratio': {
            'mean': 0.0301, 'std': 0.0095, 'p10': 0.0179, 'p50': 0.0297, 'p90': 0.0431
        },
        'volume_balance_ratio': {
            'mean': 0.327, 'std': 0.427, 'p10': -0.201, 'p50': 0.301, 'p90': 0.873
        },
        'heating_ave_power_to_building_MW': {
            'mean': 14.42, 'std': 495.97, 'p10': 0.14, 'p50': 10.22, 'p90': 23.81
        },
        'cooling_ave_power_to_building_MW': {
            'mean': 16.18, 'std': 12.81, 'p10': 2.17, 'p50': 14.57, 'p90': 32.63
        },
        'heating_annual_energy_building_GWhth': {
            'mean': 72.58, 'std': 2606.83, 'p10': 0.64, 'p50': 48.80, 'p90': 116.58
        },
        'cooling_annual_energy_building_GWhth': {
            'mean': 40.87, 'std': 30.96, 'p10': 5.63, 'p50': 37.41, 'p90': 81.06
        },
        'heating_monthly_to_building': {
            'mean': 10.73, 'std': 369.00, 'p10': 0.11, 'p50': 7.61, 'p90': 17.71
        },
        'cooling_monthly_to_building': {
            'mean': 12.04, 'std': 9.53, 'p10': 1.62, 'p50': 10.84, 'p90': 24.28
        },
        'heating_annual_elec_energy_GWhe': {
            'mean': 35.24, 'std': 2606.44, 'p10': 0.71, 'p50': 12.85, 'p90': 44.08
        },
        'cooling_annual_elec_energy_GWhe': {
            'mean': 0.75, 'std': 11.74, 'p10': 0.06, 'p50': 0.52, 'p90': 1.84
        },
        'heating_system_cop': {
            'mean': 3.50, 'std': 8.97, 'p10': 1.41, 'p50': 3.41, 'p90': 5.35
        },
        'cooling_system_cop': {
            'mean': 72.78, 'std': 26.40, 'p10': 38.98, 'p50': 72.44, 'p90': 106.25
        },
        'heating_elec_energy_per_thermal': {
            'mean': 0.320, 'std': 4.997, 'p10': 0.179, 'p50': 0.287, 'p90': 0.621
        },
        'cooling_elec_energy_per_thermal': {
            'mean': 0.0161, 'std': 0.0850, 'p10': 0.0094, 'p50': 0.0138, 'p90': 0.0256
        },
        'heating_co2_emissions_per_thermal': {
            'mean': 56.59, 'std': 886.82, 'p10': 27.63, 'p50': 51.42, 'p90': 116.69
        },
        'cooling_co2_emissions_per_thermal': {
            'mean': 2.95, 'std': 10.28, 'p10': 1.48, 'p50': 2.46, 'p90': 4.83
        }
    }


    # Case 5 sensitivity data
    case5_sensitivity_data = {
        'heating_ave_power_to_building_MW': {
            'aquifer_temp': 0.66,
            'heating_ave_injection_temp': -0.39,
            'heating_target_avg_flowrate_pd': 0.28,
            'heating_number_of_doublets': 0.11,
            'cop_param_c': -0.06
        },
        'heating_annual_energy_building_GWhth': {
            'aquifer_temp': 0.66,
            'heating_ave_injection_temp': -0.39,
            'heating_target_avg_flowrate_pd': 0.28,
            'heating_months': 0.14,
            'heating_number_of_doublets': 0.11
        },
        'heating_system_cop': {
            'cop_param_d': 0.60,
            'cop_param_c': -0.54,
            'heating_temp_to_building': -0.29,
            'aquifer_temp': 0.19,
            'cop_param_b': -0.10
        },
        'cooling_system_cop': {
            'aquifer_temp': -0.82,
            'cop_param_d': -0.38,
            'cooling_ave_injection_temp': 0.30,
            'thermal_recovery_factor': 0.15,
            'water_density': 0.07
        },
        'energy_balance_ratio': {
            'tolerance_in_energy_balance': 0.81,
            'thermal_recovery_factor': -0.55,
            'heating_months': 0.02,
            'heating_temp_to_building': -0.02,
            'cooling_ave_injection_temp': 0.01
        },
        'volume_balance_ratio': {
            'aquifer_temp': -0.91,
            'heating_ave_injection_temp': 0.36,
            'cooling_ave_injection_temp': 0.12,
            'tolerance_in_energy_balance': -0.03,
            'cop_param_d': -0.02
        },
        'cooling_ave_power_to_building_MW': {
            'aquifer_temp': 0.74,
            'heating_ave_injection_temp': -0.43,
            'heating_target_avg_flowrate_pd': 0.31,
            'cooling_months': -0.21,
            'heating_months': 0.16
        },
        'cooling_annual_energy_building_GWhth': {
            'aquifer_temp': 0.76,
            'heating_ave_injection_temp': -0.44,
            'heating_target_avg_flowrate_pd': 0.32,
            'heating_months': 0.17,
            'heating_number_of_doublets': 0.13
        }
    }
    
    # Case 4: Complex mixed distributions
    cases.append(ValidationCase(
        name="Case 4",
        description="Complex mixed distributions validation",
        iterations=10000,
        seed=888,
        distributions=complex_mixed_distributions,
        reference_data=complex_mixed_reference_data,
        sensitivity_data=case4_sensitivity_data
    ))
    
    # Case 5: Comprehensive mixed distributions
    cases.append(ValidationCase(
        name="Case 5",
        description="Comprehensive mixed distributions validation (19 parameters)",
        iterations=10000,
        seed=888,
        distributions=comprehensive_mixed_distributions,
        reference_data=comprehensive_reference_data,
        sensitivity_data=case5_sensitivity_data
    ))
    
    return cases


def run_comprehensive_validation_suite(output_dir: str = "validation_results") -> Dict[str, Any]:
    """
    Run the complete validation suite with all predefined test cases.
    
    Args:
        output_dir: Directory to save validation results
        
    Returns:
        Dictionary containing all validation results
    """
    print("ATES Monte Carlo Validation Framework")
    print("=" * 60)
    print("Academic validation tool for comparing Python ATES implementation")
    print("with Crystal Ball results across multiple test cases.")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize framework
    framework = ATESValidationFramework()
    
    # Get predefined test cases
    validation_cases = create_predefined_validation_cases()
    
    print(f"\nRunning {len(validation_cases)} validation cases...")
    
    # Track overall performance
    start_time = time.time()
    case_timings = {}
    
    # Run each validation case
    for i, case in enumerate(validation_cases, 1):
        print(f"\n[{i}/{len(validation_cases)}] Running {case.name}...")
        case_start = time.time()
        
        try:
            results = framework.run_validation_case(case, verbose=True)
            case_end = time.time()
            case_timings[case.name] = case_end - case_start
            
            if 'error' not in results:
                print(f"âœ“ {case.name} completed successfully in {case_timings[case.name]:.1f}s")
            else:
                print(f"âœ— {case.name} failed: {results['error']}")
                
        except Exception as e:
            case_end = time.time()
            case_timings[case.name] = case_end - case_start
            print(f"âœ— {case.name} failed with exception: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    
    # Export comprehensive report
    report_path = output_path / "comprehensive_validation_report.json"
    framework.export_validation_report(str(report_path))
    
    # Generate summary
    print("\n" + "=" * 80)
    print("VALIDATION SUITE SUMMARY")
    print("=" * 80)
    
    total_cases = len(framework.validation_results)
    successful_cases = sum(1 for result in framework.validation_results.values() 
                          if 'error' not in result)
    
    print(f"Total cases run: {total_cases}")
    print(f"Successful cases: {successful_cases}")
    print(f"Success rate: {successful_cases/total_cases*100:.1f}%")
    print(f"Total runtime: {total_time:.1f} seconds")
    
    # Show case timings
    print(f"\nCase Performance:")
    for case_name, timing in case_timings.items():
        status = "âœ“" if case_name in framework.validation_results and 'error' not in framework.validation_results[case_name] else "âœ—"
        print(f"  {status} {case_name}: {timing:.1f}s")
    
    if successful_cases > 0:
        print(f"\nDetailed results exported to: {output_path.absolute()}")
        print(f"Files generated:")
        print(f"  - {report_path.name}: Comprehensive JSON report")
    
    print("\nValidation suite completed!")
    
    return framework.validation_results


def run_single_case_validation(case_name: str, custom_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run validation for a single predefined case with optional custom configuration.
    
    Args:
        case_name: Name of the predefined case to run ("Case 1", "Case 2", etc.)
        custom_config: Optional dictionary to override case parameters
        
    Returns:
        Dictionary containing validation results
    """
    # Get predefined cases
    cases = create_predefined_validation_cases()
    
    # Find the requested case
    target_case = None
    for case in cases:
        if case.name == case_name:
            target_case = case
            break
    
    if target_case is None:
        available_cases = [case.name for case in cases]
        raise ValueError(f"Case '{case_name}' not found. Available cases: {available_cases}")
    
    # Apply custom configuration if provided
    if custom_config:
        if 'iterations' in custom_config:
            target_case.iterations = custom_config['iterations']
        if 'seed' in custom_config:
            target_case.seed = custom_config['seed']
        if 'distributions' in custom_config:
            target_case.distributions.update(custom_config['distributions'])
    
    # Run the validation
    framework = ATESValidationFramework()
    results = framework.run_validation_case(target_case, verbose=True)
    
    return results

def main():
    """
    Main entry point for the validation command.
    
    This function handles command line arguments and runs the appropriate validation.
    """
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            # Quick validation using Case 1
            print("Running quick validation (Case 1)...")
            result = run_single_case_validation("Case 1")
            if 'error' not in result:
                print("âœ“ Quick validation completed successfully")
            else:
                print(f"âœ— Quick validation failed: {result['error']}")
        
        elif command.startswith("case"):
            # Run specific case
            try:
                case_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
                case_name = f"Case {case_num}"
                print(f"Running {case_name}...")
                result = run_single_case_validation(case_name)
                if 'error' not in result:
                    print(f"âœ“ {case_name} completed successfully")
                else:
                    print(f"âœ— {case_name} failed: {result['error']}")
            except (ValueError, IndexError):
                print("Invalid case number. Use: ates-validate case <number>")
        
        elif command == "help":
            print("ATES Validation Framework")
            print("Usage:")
            print("  ates-validate          # Run full suite")
            print("  ates-validate quick    # Run Case 1 only")
            print("  ates-validate case <n> # Run specific case")
            print("  ates-validate help     # Show this help")
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'ates-validate help' for usage information")
    
    else:
        # Default: Run comprehensive validation suite
        try:
            run_comprehensive_validation_suite()
        except KeyboardInterrupt:
            print("\n\nValidation interrupted by user.")
        except Exception as e:
            print(f"\n\nValidation failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    """
    Main entry point for running the validation framework.
    
    Usage:
        python validation_framework.py
        python validation_framework.py quick      # Run Case 1 only
        python validation_framework.py case <n>   # Run specific case
        python validation_framework.py help       # Show help
    
    This will run all predefined validation cases and export results.
    """
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            # Quick validation using Case 1
            print("Running quick validation (Case 1)...")
            result = run_single_case_validation("Case 1")
            if 'error' not in result:
                print("âœ“ Quick validation completed successfully")
            else:
                print(f"âœ— Quick validation failed: {result['error']}")
        
        elif command.startswith("case"):
            # Extract case number from "case5" format
            try:
                if len(command) > 4 and command[4:].isdigit():
                    case_num = int(command[4:])
                else:
                    case_num = 1
                case_name = f"Case {case_num}"
                print(f"Running {case_name}...")
                result = run_single_case_validation(case_name)
                if 'error' not in result:
                    print(f"✓ {case_name} completed successfully")
                else:
                    print(f"✗ {case_name} failed: {result['error']}")
            except (ValueError, IndexError):
                print("Invalid case number. Use: ates-validate case<number>")
        elif command == "help":
            print("ATES Validation Framework")
            print("Usage:")
            print("  python validation_framework.py          # Run full suite")
            print("  python validation_framework.py quick    # Run Case 1 only")
            print("  python validation_framework.py case <n> # Run specific case")
            print("  python validation_framework.py help     # Show this help")
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'python validation_framework.py help' for usage information")
    
    else:
        # Run comprehensive validation suite
        try:
            run_comprehensive_validation_suite()
        except KeyboardInterrupt:
            print("\n\nValidation interrupted by user.")
        except Exception as e:
            print(f"\n\nValidation failed with error: {str(e)}")
            import traceback
            traceback.print_exc()