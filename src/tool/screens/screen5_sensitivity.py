import streamlit as st
from tool.core.visualization_module import ATESVisualizer

def main():
    st.title("Sensitivity Analysis")
    
    if st.session_state.sensitivity_results is None:
        return
    
    visualizer = ATESVisualizer(
        st.session_state.monte_carlo_results, 
        st.session_state.sensitivity_results
    )
    
    visualizer.render_sensitivity_analysis()