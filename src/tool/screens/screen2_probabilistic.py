"""
Screen 2 - Probabilistic Analysis Setup 
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from typing import Dict, List, Any, Optional
import time

from tool.core.ates_calculator import ATESParameters
from tool.core.monte_carlo_engine import ATESMonteCarloEngine, MonteCarloConfig, create_progress_callback
from tool.utils.state_management import get_app_state

def initialize_probabilistic_session_state():
    """Initialize session state for probabilistic analysis with robust initialization"""
    if 'ates_params' not in st.session_state:
        st.session_state.ates_params = ATESParameters()
    
    if 'param_distributions' not in st.session_state:
        try:
            st.session_state.param_distributions = initialize_distributions()
        except Exception as e:
            st.warning(f"Distribution initialization warning: {e}")
            st.session_state.param_distributions = {}
            initialize_distributions_from_ates_params()
    
    if 'monte_carlo_results' not in st.session_state:
        st.session_state.monte_carlo_results = None
    
    if 'monte_carlo_iterations' not in st.session_state:
        st.session_state.monte_carlo_iterations = 10000
    
    if 'sensitivity_results' not in st.session_state:
        st.session_state.sensitivity_results = None
    
    if 'mc_config' not in st.session_state:
        st.session_state.mc_config = MonteCarloConfig()
    
    if 'param_config_version' not in st.session_state:
        st.session_state.param_config_version = 0
    
    if 'stable_param_values' not in st.session_state:
        st.session_state.stable_param_values = {}

def initialize_distributions_from_ates_params() -> None:
    """Initialize distributions directly from ATES parameters"""
    params = st.session_state.ates_params
    distributions = {}
    
    probabilistic_params = [
        'aquifer_temp', 'water_density', 'water_specific_heat_capacity',
        'thermal_recovery_factor', 'heating_target_avg_flowrate_pd',
        'tolerance_in_energy_balance', 'heating_number_of_doublets',
        'heating_months', 'cooling_months', 'pump_energy_density',
        'heating_ave_injection_temp', 'heating_temp_to_building',
        'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d',
        'carbon_intensity', 'cooling_ave_injection_temp', 'cooling_temp_to_building'
    ]
    
    for param_name in probabilistic_params:
        if hasattr(params, param_name):
            current_value = getattr(params, param_name)
            distributions[param_name] = {
                'type': 'single_value',
                'value': current_value,
                'min': current_value * 0.8,
                'max': current_value * 1.2,
                'most_likely': current_value,
                'mean': current_value,
                'std': max(current_value * 0.1, 0.01),
                'location': 0.0,  
                'use_log_params': False,  
            }
    
    st.session_state.param_distributions = distributions

def initialize_distributions() -> Dict[str, Dict[str, Any]]:
    """Initialize parameter distribution configurations"""
    params = ATESParameters()
    distributions = {}
    
    probabilistic_params = [
        'aquifer_temp', 'water_density', 'water_specific_heat_capacity',
        'thermal_recovery_factor', 'heating_target_avg_flowrate_pd',
        'tolerance_in_energy_balance', 'heating_number_of_doublets',
        'heating_months', 'cooling_months', 'pump_energy_density',
        'heating_ave_injection_temp', 'heating_temp_to_building',
        'cop_param_a', 'cop_param_b', 'cop_param_c', 'cop_param_d',
        'carbon_intensity', 'cooling_ave_injection_temp', 'cooling_temp_to_building'
    ]
    
    for param_name in probabilistic_params:
        if hasattr(params, param_name):
            current_value = getattr(params, param_name)
            distributions[param_name] = {
                'type': 'single_value',
                'value': current_value,
                'min': current_value * 0.8,
                'max': current_value * 1.2,
                'most_likely': current_value,
                'mean': current_value,
                'std': max(current_value * 0.1, 0.01),
                'location': 0.0,  
                'use_log_params': False,  
            }
    
    return distributions

def sync_from_deterministic():
    """sync logic"""
    updated_params = []
    
    for param_name in st.session_state.param_distributions:
        if hasattr(st.session_state.ates_params, param_name):
            current_value = getattr(st.session_state.ates_params, param_name)
            dist = st.session_state.param_distributions[param_name]
            
          
            if dist['type'] == 'single_value':
                old_value = dist.get('value', 0)
            elif dist['type'] == 'triangular':
                old_value = dist.get('most_likely', 0)
            elif dist['type'] in ['normal', 'lognormal']:
                old_value = dist.get('mean', 0)
            elif dist['type'] == 'range':
                old_value = (dist.get('min', 0) + dist.get('max', 0)) / 2
            else:
                old_value = dist.get('value', 0)
            
            if abs(old_value - current_value) > 1e-6:  
                updated_params.append(param_name)
                
                
                if dist['type'] == 'single_value':
                    dist['value'] = current_value
                elif dist['type'] == 'triangular':
                   
                    old_range = dist.get('max', 0) - dist.get('min', 0)
                    if old_range > 0:
                        ratio = old_range / old_value if old_value != 0 else 0.4
                        new_range = current_value * ratio
                        dist['min'] = current_value - new_range / 2
                        dist['max'] = current_value + new_range / 2
                    else:
                        dist['min'] = current_value * 0.8
                        dist['max'] = current_value * 1.2
                    dist['most_likely'] = current_value
                elif dist['type'] in ['normal', 'lognormal']:
                   
                    old_std = dist.get('std', 0)
                    if old_value != 0:
                        std_ratio = old_std / old_value
                        dist['std'] = current_value * std_ratio
                    else:
                        dist['std'] = max(current_value * 0.1, 0.01)
                    dist['mean'] = current_value
                elif dist['type'] == 'range':
                    
                    old_min = dist.get('min', 0)
                    old_max = dist.get('max', 0)
                    if old_value != 0:
                        min_ratio = old_min / old_value
                        max_ratio = old_max / old_value
                        dist['min'] = current_value * min_ratio
                        dist['max'] = current_value * max_ratio
                    else:
                        dist['min'] = current_value * 0.9
                        dist['max'] = current_value * 1.1
                
                
                dist['value'] = current_value
                dist['mean'] = current_value
                dist['most_likely'] = current_value
    
    if updated_params:
        
        st.session_state.stable_param_values = {}
        st.session_state.param_config_version += 1
        st.success(f"Synchronized {len(updated_params)} parameters from Quick Look")

def sync_to_deterministic():
    """
    Sync parameter values from probabilistic setup to deterministic calculation
    """
    updated_count = 0
    
    for param_name, dist in st.session_state.param_distributions.items():
        if hasattr(st.session_state.ates_params, param_name):
            if dist['type'] == 'single_value':
                new_value = dist['value']
            elif dist['type'] == 'triangular':
                new_value = dist['most_likely']
            elif dist['type'] in ['normal', 'lognormal']:
                new_value = dist['mean']
            else:  # range
                new_value = (dist['min'] + dist['max']) / 2
            
            current_value = getattr(st.session_state.ates_params, param_name)
            if abs(current_value - new_value) > 1e-6:
                setattr(st.session_state.ates_params, param_name, new_value)
                updated_count += 1
    
    if updated_count > 0:
        st.session_state.ates_params.__post_init__()

def render_parameter_config(param_name: str, param_label: str):
    """
    Render parameter configuration interface 
    """
    dist_config = st.session_state.param_distributions[param_name]
    current_type = dist_config.get('type', 'single_value')
    is_uncertain = current_type != 'single_value'
    version = st.session_state.get('param_config_version', 0)
    
    with st.expander(f"{param_label}", expanded=is_uncertain):
        type_key = f"type_{param_name}_v{version}"
        supported_types = ['single_value', 'range', 'triangular', 'normal', 'lognormal']
        
        new_dist_type = st.selectbox(
            "Parameter Type",
            supported_types,
            index=supported_types.index(current_type),
            key=type_key,
            format_func=lambda x: {
                'single_value': 'Fixed Value (Deterministic)',
                'range': 'Uniform Distribution',
                'triangular': 'Triangular Distribution',
                'normal': 'Normal Distribution',
                'lognormal': 'Log-Normal Distribution'
            }[x]
        )
        
        
        if new_dist_type != current_type:
            
            if current_type == 'single_value':
                center_value = dist_config.get('value', 14.0)
            elif current_type == 'triangular':
                center_value = dist_config.get('most_likely', 14.0)
            elif current_type in ['normal', 'lognormal']:
                center_value = dist_config.get('mean', 14.0)
            elif current_type == 'range':
                center_value = (dist_config.get('min', 14.0) + dist_config.get('max', 14.0)) / 2
            else:
                center_value = dist_config.get('value', 14.0)
            
           
            dist_config['type'] = new_dist_type
            
            if new_dist_type == 'single_value':
                dist_config['value'] = center_value
            elif new_dist_type == 'range':
                
                dist_config['min'] = center_value * 0.9
                dist_config['max'] = center_value * 1.1
            elif new_dist_type == 'triangular':
                dist_config['min'] = center_value * 0.8
                dist_config['most_likely'] = center_value
                dist_config['max'] = center_value * 1.2
            elif new_dist_type in ['normal', 'lognormal']:
                dist_config['mean'] = center_value
                dist_config['std'] = max(center_value * 0.1, 0.01)
                if new_dist_type == 'lognormal':
                    dist_config['location'] = 0.0
                    dist_config['use_log_params'] = False
            
            from tool.utils.state_management import mark_case_modified
            mark_case_modified()
            st.session_state.param_config_version = version + 1
            
            
            keys_to_remove = [k for k in st.session_state.stable_param_values.keys() 
                             if k.startswith(f"{param_name}_v")]
            for k in keys_to_remove:
                del st.session_state.stable_param_values[k]
            
            st.rerun()
        
        render_distribution_params_stable(param_name, dist_config, new_dist_type, version)
        
        if new_dist_type != 'single_value':
            st.markdown("---")
            render_distribution_preview(param_name, dist_config, param_label)

def render_distribution_params_stable(param_name: str, dist_config: Dict, dist_type: str, version: int):
    """
    Render distribution specific parameters 
    """
    stable_key = f"{param_name}_v{version}"
    
   
    if stable_key not in st.session_state.stable_param_values:
        st.session_state.stable_param_values[stable_key] = dist_config.copy()
    
    stable_config = st.session_state.stable_param_values[stable_key]
    
    
    for key in dist_config:
        if key not in stable_config or stable_config[key] != dist_config[key]:
            stable_config[key] = dist_config[key]
    
    def update_stable_config(key: str, value: Any):
        """
        Callback function to update stable configuration
        """
        stable_config[key] = value
        dist_config[key] = value
        from tool.utils.state_management import mark_case_modified
        mark_case_modified()
    
    if dist_type == 'single_value':
        val_key = f"val_{param_name}_v{version}"
        st.number_input(
            "Value",
            value=float(stable_config.get('value', 0)),
            key=val_key,
            format="%.4f",
            on_change=lambda: update_stable_config('value', st.session_state[val_key])
        )
    
    elif dist_type == 'range':
        col1, col2 = st.columns(2)
        with col1:
            min_key = f"min_{param_name}_v{version}"
            st.number_input(
                "Minimum",
                value=float(stable_config.get('min', 0)),
                key=min_key,
                format="%.4f",
                on_change=lambda: update_stable_config('min', st.session_state[min_key])
            )
                
        with col2:
            max_key = f"max_{param_name}_v{version}"
            st.number_input(
                "Maximum",
                value=float(stable_config.get('max', 1)),
                key=max_key,
                format="%.4f",
                on_change=lambda: update_stable_config('max', st.session_state[max_key])
            )
        
        if stable_config.get('min', 0) >= stable_config.get('max', 1):
            st.error("Minimum must be less than maximum")
    
    elif dist_type == 'triangular':
        col1, col2, col3 = st.columns(3)
        with col1:
            tri_min_key = f"tri_min_{param_name}_v{version}"
            st.number_input(
                "Minimum",
                value=float(stable_config.get('min', 0)),
                key=tri_min_key,
                format="%.4f",
                on_change=lambda: update_stable_config('min', st.session_state[tri_min_key])
            )
                
        with col2:
            tri_ml_key = f"tri_ml_{param_name}_v{version}"
            st.number_input(
                "Most Likely",
                value=float(stable_config.get('most_likely', 0.5)),
                key=tri_ml_key,
                format="%.4f",
                on_change=lambda: update_stable_config('most_likely', st.session_state[tri_ml_key])
            )
                
        with col3:
            tri_max_key = f"tri_max_{param_name}_v{version}"
            st.number_input(
                "Maximum",
                value=float(stable_config.get('max', 1)),
                key=tri_max_key,
                format="%.4f",
                on_change=lambda: update_stable_config('max', st.session_state[tri_max_key])
            )
        
        min_val = stable_config.get('min', 0)
        ml_val = stable_config.get('most_likely', 0.5)
        max_val = stable_config.get('max', 1)
        if not (min_val <= ml_val <= max_val):
            st.error("Most likely value must be between minimum and maximum")
    
    elif dist_type in ['normal', 'lognormal']:
        col1, col2 = st.columns(2)
        with col1:
            mean_key = f"mean_{param_name}_v{version}"
            st.number_input(
                "Mean",
                value=float(stable_config.get('mean', 0)),
                key=mean_key,
                format="%.4f",
                on_change=lambda: update_stable_config('mean', st.session_state[mean_key])
            )
                
        with col2:
            std_key = f"std_{param_name}_v{version}"
            st.number_input(
                "Standard Deviation",
                value=float(stable_config.get('std', 0.1)),
                min_value=0.0,
                key=std_key,
                format="%.4f",
                on_change=lambda: update_stable_config('std', st.session_state[std_key])
            )
        
        if dist_type == 'lognormal':
            col3, col4 = st.columns(2)
            with col3:
                location_key = f"location_{param_name}_v{version}"
                st.number_input(
                    "Location Parameter",
                    value=float(stable_config.get('location', 0.0)),
                    key=location_key,
                    format="%.4f",
                    help="Minimum possible value (location parameter)",
                    on_change=lambda: update_stable_config('location', st.session_state[location_key])
                )
                    
            with col4:
                use_log_key = f"use_log_{param_name}_v{version}"
                st.checkbox(
                    "Use Log Parameters",
                    value=bool(stable_config.get('use_log_params', False)),
                    key=use_log_key,
                    help="Check if mean/std are already in log space",
                    on_change=lambda: update_stable_config('use_log_params', st.session_state[use_log_key])
                )
            
            mean_val = stable_config.get('mean', 0)
            location_val = stable_config.get('location', 0.0)
            if mean_val <= location_val:
                st.error("Mean must be greater than location parameter for lognormal distribution")

def render_parameter_groups_tabs():
    """
    Render parameter configuration in organized tabs
    """
    tab1, tab2, tab3, tab4 = st.tabs([
        "Physical Parameters", 
        "Operational Parameters", 
        "COP Parameters", 
        "Cooling Parameters"
    ])
    
    with tab1:
        render_physical_parameters()
    
    with tab2:
        render_operational_parameters()
    
    with tab3:
        render_cop_parameters()
    
    with tab4:
        render_cooling_parameters()

def render_physical_parameters():
    """
    Render physical parameters section
    """
    st.subheader("Basic Physical Parameters")
    
    physical_params = [
        'aquifer_temp', 
        'water_density', 
        'water_specific_heat_capacity',
        'thermal_recovery_factor'
    ]
    
    param_labels = {
        'aquifer_temp': 'Aquifer Temperature (°C)',
        'water_density': 'Water Density (kg/m³)',
        'water_specific_heat_capacity': 'Water Specific Heat Capacity (J/kg/K)',
        'thermal_recovery_factor': 'Thermal Recovery Factor (-)'
    }
    
    for param in physical_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])

def render_operational_parameters():
    """Render operational parameters section"""
    st.subheader("System Operational Parameters")
    
    operational_params = [
        'heating_target_avg_flowrate_pd',
        'tolerance_in_energy_balance',
        'heating_number_of_doublets',
        'heating_months', 
        'cooling_months', 
        'pump_energy_density',
        'heating_ave_injection_temp', 
        'heating_temp_to_building'
    ]
    
    param_labels = {
        'heating_target_avg_flowrate_pd': 'Target Flow Rate Heating (m³/hr)',
        'tolerance_in_energy_balance': 'Energy Balance Tolerance (-)',
        'heating_number_of_doublets': 'Number of Doublets (-)',
        'heating_months': 'Heating Months',
        'cooling_months': 'Cooling Months',
        'pump_energy_density': 'Pump Energy Density (kJ/m³)',
        'heating_ave_injection_temp': 'Heating Injection Temperature (°C)',
        'heating_temp_to_building': 'Building Heating Temperature (°C)'
    }
    
    for param in operational_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])

def render_cop_parameters():
    """Render COP parameters section"""
    st.subheader("Heat Pump COP Parameters")
    
    cop_params = [
        'cop_param_a', 
        'cop_param_b', 
        'cop_param_c', 
        'cop_param_d', 
        'carbon_intensity'
    ]
    
    param_labels = {
        'cop_param_a': 'COP Parameter A (-)',
        'cop_param_b': 'COP Parameter B (-)',
        'cop_param_c': 'COP Parameter C (-)',
        'cop_param_d': 'COP Parameter D (-)',
        'carbon_intensity': 'Carbon Intensity (gCO₂/kWh)'
    }
    
    for param in cop_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])

def render_cooling_parameters():
    """Render cooling parameters section"""
    st.subheader("Cooling System Parameters")
    
    cooling_params = [
        'cooling_ave_injection_temp', 
        'cooling_temp_to_building'
    ]
    
    param_labels = {
        'cooling_ave_injection_temp': 'Cooling Injection Temperature (°C)',
        'cooling_temp_to_building': 'Building Cooling Temperature (°C)'
    }
    
    for param in cooling_params:
        if param in st.session_state.param_distributions:
            render_parameter_config(param, param_labels[param])

def render_distribution_preview(param_name: str, dist_config: Dict, param_label: str):
    """
    Render a preview of the parameter distribution
    """
    try:
        n_samples = 1000
        rng = np.random.default_rng(42)
        
        if dist_config['type'] == 'single_value':
            samples = np.full(n_samples, dist_config['value'])
        elif dist_config['type'] == 'range':
            samples = rng.uniform(dist_config['min'], dist_config['max'], n_samples)
        elif dist_config['type'] == 'triangular':
            c = (dist_config['most_likely'] - dist_config['min']) / (dist_config['max'] - dist_config['min'])
            samples = stats.triang.rvs(c, loc=dist_config['min'], 
                                     scale=dist_config['max'] - dist_config['min'], 
                                     size=n_samples, random_state=rng)
        elif dist_config['type'] == 'normal':
            samples = rng.normal(dist_config['mean'], dist_config['std'], n_samples)
        elif dist_config['type'] == 'lognormal':
            if dist_config['mean'] > 0:
                mu = np.log(dist_config['mean'])
                sigma = dist_config['std'] / dist_config['mean']
                samples = rng.lognormal(mu, sigma, n_samples)
            else:
                samples = np.full(n_samples, 0)
        else:
            samples = np.full(n_samples, 0)
        
        fig = px.histogram(
            x=samples,
            nbins=30,
            title=f"Distribution Preview: {param_label}",
            labels={'x': param_label, 'y': 'Frequency'}
        )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(samples):.3f}")
        with col2:
            st.metric("Std Dev", f"{np.std(samples):.3f}")
        with col3:
            st.metric("Min", f"{np.min(samples):.3f}")
        with col4:
            st.metric("Max", f"{np.max(samples):.3f}")
    
    except Exception as e:
        st.error(f"Error generating preview: {str(e)}")

def render_monte_carlo_settings():
    """
    Render Monte Carlo simulation settings
    """
    st.subheader("Monte Carlo Simulation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.monte_carlo_iterations = st.number_input(
            "Number of Iterations",
            value=st.session_state.monte_carlo_iterations,
            min_value=1000,
            max_value=100000,
            step=1000,
            help="More iterations = more accurate results but longer computation time"
        )
    
    with col2:
        seed = st.number_input(
            "Random Seed",
            value=42,
            min_value=0,
            help="Set for reproducible results (0 for random)"
        )
        st.session_state.mc_config.seed = seed if seed > 0 else None
    
    with col3:
        parallel = st.checkbox(
            "Parallel Processing",
            value=True,
            help="Use multiple CPU cores for faster computation"
        )
        st.session_state.mc_config.parallel = parallel
    
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_workers = st.number_input(
                "Max Workers",
                value=4,
                min_value=1,
                max_value=16,
                help="Number of parallel workers (if parallel processing enabled)"
            )
            st.session_state.mc_config.max_workers = max_workers
        
        with col2:
            chunk_size = st.number_input(
                "Chunk Size",
                value=1000,
                min_value=100,
                max_value=10000,
                help="Number of iterations per chunk for progress tracking"
            )
            st.session_state.mc_config.chunk_size = chunk_size
    
    st.session_state.mc_config.iterations = st.session_state.monte_carlo_iterations

def render_enabled_parameters_summary():
    """
    Render parameter configuration summary with correct uncertainty counting
    """
    uncertain_params = []
    for name, dist in st.session_state.param_distributions.items():
        if dist['type'] != 'single_value':
            uncertain_params.append(name)
    
    st.subheader("Parameter Configuration Summary")
    
    if not uncertain_params:
        st.info("All parameters use fixed values (deterministic analysis)")
        return
    
    st.success(f"{len(uncertain_params)} parameters configured with uncertainty")
    
    summary_data = []
    for param in uncertain_params:
        dist = st.session_state.param_distributions[param]
        
        if dist['type'] == 'range':
            range_info = f"Range: [{dist['min']:.3f}, {dist['max']:.3f}]"
        elif dist['type'] == 'triangular':
            range_info = f"Triangular: [{dist['min']:.3f}, {dist['most_likely']:.3f}, {dist['max']:.3f}]"
        elif dist['type'] == 'normal':
            range_info = f"Normal: μ={dist['mean']:.3f}, σ={dist['std']:.3f}"
        elif dist['type'] == 'lognormal':
            range_info = f"Log-Normal: μ={dist['mean']:.3f}, σ={dist['std']:.3f}"
        else:
            range_info = "Unknown"
        
        summary_data.append({
            'Parameter': param.replace('_', ' ').title(),
            'Distribution': dist['type'].replace('_', ' ').title(),
            'Range/Parameters': range_info
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

def render_monte_carlo_execution():
    """
    Render Monte Carlo execution interface with corrected parameter counting
    """
    uncertain_params = []
    for name, dist in st.session_state.param_distributions.items():
        if dist['type'] != 'single_value':
            uncertain_params.append(name)

    validation_errors = validate_distribution_config()
    if validation_errors:
        st.error("Configuration errors:")
        for error in validation_errors:
            st.error(f"• {error}")
        return
    
    if st.button("Run Analysis", type="primary", use_container_width=True):
        run_monte_carlo_analysis()

def validate_distribution_config() -> List[str]:
    """
    Validate distribution configurations
    """
    errors = []
    
    for param_name, dist in st.session_state.param_distributions.items():
        dist_type = dist['type']
        
        if dist_type == 'range':
            if dist['min'] >= dist['max']:
                errors.append(f"{param_name}: Minimum must be less than maximum")
        
        elif dist_type == 'triangular':
            if not (dist['min'] <= dist['most_likely'] <= dist['max']):
                errors.append(f"{param_name}: Most likely value must be between min and max")
        
        elif dist_type in ['normal', 'lognormal']:
            if dist['std'] <= 0:
                errors.append(f"{param_name}: Standard deviation must be positive")
            
            if dist_type == 'lognormal' and dist['mean'] <= 0:
                errors.append(f"{param_name}: Mean must be positive for log-normal distribution")
    
    return errors

def run_monte_carlo_analysis():
    """
    Execute Monte Carlo analysis with parameter samples saved
    """
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # creating a monte carlo engine
        mc_engine = ATESMonteCarloEngine(
            st.session_state.ates_params,
            st.session_state.mc_config
        )
        
        # creating a progress callback
        progress_callback = create_progress_callback(progress_bar, status_text)
        
        # generating parameter samples and saving
        rng = np.random.default_rng(st.session_state.mc_config.seed)
        parameter_samples = mc_engine._generate_parameter_samples(
            st.session_state.param_distributions, rng
        )
        
        # saving parameter samples to session state
        st.session_state.parameter_samples = parameter_samples
        
        # recording start time
        start_time = time.time()
        
        # run simulation
        results_df = mc_engine.run_simulation(
            st.session_state.param_distributions,
            progress_callback
        )
        
        # calculate computation time
        computation_time = time.time() - start_time
        
        # storgae result
        st.session_state.monte_carlo_results = results_df
        st.session_state._last_mc_computation_time = computation_time
        
        # calculate sensitivity analysis
        uncertain_params = {name: config for name, config in st.session_state.param_distributions.items() 
                           if config['type'] != 'single_value'}
        
        if uncertain_params:
            try:
                sensitivity_results = mc_engine.calculate_sensitivity_analysis(parameter_samples)
                st.session_state.sensitivity_results = sensitivity_results
            except Exception as e:
                st.warning(f"Sensitivity analysis failed: {str(e)}")
                st.session_state.sensitivity_results = None
        else:
            st.session_state.sensitivity_results = None

        # finish
        st.session_state._mc_completed = True
        
        # clean progress
        progress_bar.empty()
        status_text.empty()
        
        # show success info
        st.success("Monte Carlo analysis completed!")

        st.rerun()

    except Exception as e:
        st.error(f"Monte Carlo analysis failed: {str(e)}")
        # clean progress
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def display_monte_carlo_results():
   """
   Display stored Monte Carlo results
   """
   results_df = st.session_state.monte_carlo_results
   
   # Calculate basic metrics
   successful_runs = int(results_df['success'].sum()) if 'success' in results_df.columns else len(results_df)
   success_rate = successful_runs / len(results_df) * 100
   
   # Get computation time if stored
   computation_time = st.session_state.get('_last_mc_computation_time', 0)
   
   # Show metrics
   col1, col2, col3, col4 = st.columns(4)
   with col1:
       st.metric("Total Iterations", f"{len(results_df):,}")
   with col2:
       st.metric("Successful", f"{successful_runs:,}")
   with col3:
       st.metric("Success Rate", f"{success_rate:.1f}%")
   with col4:
       st.metric("Computation Time", f"{computation_time:.1f}s")
   
   # Show quick results preview
   if successful_runs > 0:
       st.subheader("Quick Results Preview")
       
       successful_results = results_df[results_df['success'] == True] if 'success' in results_df.columns else results_df
       preview_params = ['heating_system_cop', 'cooling_system_cop', 'volume_balance_ratio', 'energy_balance_ratio', 'heating_annual_energy_building_GWhth', 'cooling_annual_energy_building_GWhth',
                         'heating_co2_emissions_per_thermal', 'cooling_co2_emissions_per_thermal', 'heating_annual_elec_energy_GWhe', 'cooling_annual_elec_energy_GWhe']
       
       preview_data = []
       for param in preview_params:
           if param in successful_results.columns:
               data = successful_results[param].dropna()
               data = data[np.isfinite(data)] 
               if len(data) > 0:
                   preview_data.append({
                       'Parameter': param.replace('_', ' ').title(),
                       'Mean': f"{data.mean():.3f}",
                       'Std': f"{data.std():.3f}",
                       'P10': f"{data.quantile(0.10):.3f}",
                       'P50': f"{data.quantile(0.50):.3f}",
                       'P90': f"{data.quantile(0.90):.3f}"
                   })
       
       if preview_data:
           preview_df = pd.DataFrame(preview_data)
           st.dataframe(preview_df, use_container_width=True, hide_index=True)

def render_monte_carlo_export():
    """
    Render Monte Carlo data export options
    """
    st.markdown("### Raw Monte Carlo Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Full Raw Data", use_container_width=True):
            results_csv = st.session_state.monte_carlo_results.to_csv(index=False, encoding='utf-8-sig')
            
            app_state = get_app_state()
            case_name = app_state.get_case_name()
            clean_case_name = app_state._clean_filename(case_name)
            
            st.download_button(
                label="Download Complete Results CSV",
                data=results_csv,
                file_name=f"{clean_case_name}_monte_carlo_raw_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_full_mc_csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("Export Key Results Only", use_container_width=True):
            
            key_columns = [
                'iteration', 'success',
                'heating_system_cop', 'cooling_system_cop', 'overall_system_cop',
                'heating_annual_energy_building_gwh', 'cooling_annual_energy_building_gwh',
                'total_annual_energy_gwh', 'total_electrical_energy_gwh',
                'heating_co2_emissions', 'cooling_co2_emissions',
                'energy_balance_ratio', 'volume_balance_ratio',
                'heating_direct_mode', 'cooling_direct_mode'
            ]
            
            available_key_columns = [col for col in key_columns 
                                   if col in st.session_state.monte_carlo_results.columns]
            
            key_results_df = st.session_state.monte_carlo_results[available_key_columns]
            key_results_csv = key_results_df.to_csv(index=False, encoding='utf-8-sig')
            
            app_state = get_app_state()
            case_name = app_state.get_case_name()
            clean_case_name = app_state._clean_filename(case_name)
            
            st.download_button(
                label="Download Key Results CSV",
                data=key_results_csv,
                file_name=f"{clean_case_name}_monte_carlo_key_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_key_mc_csv",
                use_container_width=True
            )
    
    with col3:  
        if st.button("Export Input + Output Data", use_container_width=True):
            # combine input and output
            if 'parameter_samples' in st.session_state:
                input_data = st.session_state.parameter_samples
                output_data = st.session_state.monte_carlo_results
                
                # combine results
                combined_data = pd.concat([input_data, output_data], axis=1)
                combined_csv = combined_data.to_csv(index=False, encoding='utf-8-sig')
                
                app_state = get_app_state()
                case_name = app_state.get_case_name()
                clean_case_name = app_state._clean_filename(case_name)
                
                st.download_button(
                    label="Download Input+Output CSV",
                    data=combined_csv,
                    file_name=f"{clean_case_name}_complete_data_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_complete_csv",
                    use_container_width=True
                )
            else:
                st.error("Parameter samples not available")

def reset_all_distributions():
    """
    Reset all distribution configurations to default state 
    """
    st.session_state.param_config_version = st.session_state.get('param_config_version', 0) + 1
    
    st.session_state.stable_param_values = {}
    
    st.session_state.param_distributions = initialize_distributions()
    
    # Clear all analysis results
    analysis_keys = [
        'monte_carlo_results',
        'sensitivity_results', 
        'calculation_count',
        'last_calculation_time',
        'calculation_status'
    ]
    
    for key in analysis_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.calculation_count = 0
    st.session_state.calculation_status = 'not_started'
    st.session_state.last_calculation_time = None
    
    from tool.utils.state_management import mark_case_modified
    mark_case_modified()

def main():
    """
    Main function for Screen 2 
    """
    
    # Initialize probabilistic analysis session state variables
    initialize_probabilistic_session_state()
    
    # Check if parameter distributions exist but stable values are not cached
    if st.session_state.get('param_distributions') and not st.session_state.get('stable_param_values'):
        st.session_state.param_config_version = st.session_state.get('param_config_version', 0) + 1  # Increment config version
        st.session_state.stable_param_values = {}  # Initialize empty stable values cache
    
    # Set up page header and description
    st.title("Probabilistic Analysis Setup")
    st.markdown("Configure probability distributions for uncertain parameters and run Monte Carlo analysis")
    
    # Create control buttons for synchronization and reset operations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Button to import parameter values from deterministic calculation
        if st.button("Sync FROM Quick Look", use_container_width=True,
                     help="Import parameter values from deterministic calculation"):
            sync_from_deterministic()  # Import values from Quick Look screen
            st.session_state.param_config_version = st.session_state.get('param_config_version', 0) + 1  # Update version
            st.session_state.stable_param_values = {}  # Clear cached stable values
            st.success("Synchronized from Quick Look")
            st.rerun()  # Refresh the page to reflect changes
    
    with col2:
        # Button to export representative values to deterministic calculation
        if st.button("Sync TO Quick Look", use_container_width=True,
                     help="Export representative values to deterministic calculation"):
            sync_to_deterministic()  # Export values to Quick Look screen
            st.success("Synchronized to Quick Look")
    
    with col3:
        # Button to reset all parameters to their default values
        if st.button("Reset All", use_container_width=True,
                     help="Reset all parameters to default values"):
            reset_all_distributions()  # Reset all probability distributions
            st.success("All parameters reset to defaults")
            st.rerun()  # Refresh the page to reflect changes
    
    st.markdown("---")  # Add visual separator
    
    # Create main tabs for different sections of the interface
    tab1, tab2, tab3 = st.tabs([
        "Parameter Configuration",
        "Simulation Settings", 
        "▶️ Run Analysis"
    ])
    
    with tab1:
        # Tab for configuring probability distributions
        st.markdown("### Configure Probability Distributions")
        render_parameter_groups_tabs()  # Display parameter configuration interface
        st.markdown("---")
        render_enabled_parameters_summary()  # Show summary of enabled uncertain parameters
    
    with tab2:
        # Tab for Monte Carlo simulation settings
        render_monte_carlo_settings()  # Display simulation configuration options
    
    with tab3:
        # Tab for running Monte Carlo analysis
        render_monte_carlo_execution()  # Display execution controls and progress
    
    # Display results if Monte Carlo analysis has been completed
    if st.session_state.monte_carlo_results is not None:
        st.markdown("---")
        display_monte_carlo_results()  # Show analysis results and visualizations
        
        # Provide export options for the results data
        with st.expander("Export Monte Carlo Data", expanded=False):
            render_monte_carlo_export()  # Display data export interface