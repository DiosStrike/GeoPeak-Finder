"""
Pittsburgh Urban Risk Analytics Interface
Interactive Web Application for Topographical Digital Twin Visualization and Risk Assessment.
"""

import streamlit as st
from PIL import Image

# Global configuration for the analytical interface
st.set_page_config(page_title="Pittsburgh Urban Risk Analytics Engine", layout="wide")

# --- Sidebar: Project Metadata & Methodologies ---
st.sidebar.header("Project Metadata")
st.sidebar.markdown("""
**Principal Investigator**: Tanghao Chen (Dios)  
**Academic Affiliation**: CMU AI Engineering - Civil & Environmental Engineering  

**Core Methodologies**:
* **Neural Surface Fitting**: MLP-based coordinate regression.
* **Global Optimization**: Multi-start Stochastic Gradient Ascent (SGA).
* **Geospatial Analytics**: Automatic Differentiation for slope magnitude computation.
""")

# --- Main Interface Header ---
st.title("Urban Risk Analytics Engine: Pittsburgh Topographical Digital Twin")
st.markdown("""
This analytical platform integrates Deep Learning with Geospatial Engineering to construct a **Differentiable Digital Twin** of the Pittsburgh region. 
The system facilitates quantitative environmental risk assessment through neural surface inference and gradient-based optimization algorithms.
""")

st.divider()

# --- Analytical Modules (Tabs) ---
tab1, tab2, tab3 = st.tabs([
    "Global Maximum Identification", 
    "Hydraulic Sink Analysis", 
    "Geotechnical Slope Criticality"
])

with tab1:
    st.subheader("Topographical Peak Localization via Gradient Ascent")
    try:
        image = Image.open('results/1_basic_gradient_map.png')
        st.image(image, caption="Figure 1: Geographic peak identification via neural surface optimization.")
    except FileNotFoundError:
        st.error("Asset '1_basic_gradient_map.png' not found in /results directory.")
        
    st.markdown("""
    **Technical Summary**:  
    Utilizing Stochastic Gradient Ascent (SGA) on the neural terrain surface, the engine identifies global elevation maxima. 
    The model accurately localized the **Robert Williams Reservoir** as the primary geographic peak within the study area.
    """)

with tab2:
    st.subheader("Hydraulic Sink Detection & Flooding Susceptibility")
    try:
        image = Image.open('results/2_flood_risk_analysis.png')
        st.image(image, caption="Figure 2: Spatial distribution of hydraulic sinks indicating high-intensity accumulation risk.")
    except FileNotFoundError:
        st.error("Asset '2_flood_risk_analysis.png' not found in /results directory.")

    st.markdown("""
    **Technical Summary**:  
    By executing gradient descent across the differentiable surface, the system identifies localized topographical minima (Hydraulic Sinks). 
    These points represent critical accumulation zones susceptible to urban flooding during extreme precipitation events.
    """)

with tab3:
    st.subheader("Geotechnical Risk Assessment: Slope Criticality")
    try:
        image = Image.open('results/3_landslide_risk_analysis.png')
        st.image(image, caption="Figure 3: Landslide susceptibility zones delineated via slope gradient thresholds.")
    except FileNotFoundError:
        st.error("Asset '3_landslide_risk_analysis.png' not found in /results directory.")

    st.markdown("""
    **Technical Summary**:  
    Leveraging PyTorch-based Automatic Differentiation (Autograd), the engine computes the first-order partial derivatives of the terrain surface. 
    The red contour lines delineate zones exceeding critical slope thresholds, indicating high landslide susceptibility.
    """)