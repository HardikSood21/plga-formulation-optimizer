"""
PLGA Nanoparticle Formulation Optimizer
Dr. Ruchi Chawla's Lab | IIT BHU
Developed by: Hardik Sood
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="PLGA Nanoformulation Optimizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        color: #2E86AB;
        text-align: center;
        padding: 20px 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
        padding: 10px;
        font-size: 18px;
        border-radius: 10px;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #D4EDDA;
        border: 1px solid #28A745;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #DEE2E6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<div class="main-header">🧬 PLGA Nanoparticle Formulation Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dr. Ruchi Chawla\'s Lab | IIT BHU Pharmaceutical Engineering</div>', unsafe_allow_html=True)

# ========== LOAD MODELS AND DATA ==========
@st.cache_resource
def load_models():
    """Load pre-trained models and dataset"""
    try:
        with open('model_particle_size_final.pkl', 'rb') as f:
            model_size = pickle.load(f)
        with open('model_ee_final.pkl', 'rb') as f:
            model_ee = pickle.load(f)
        df = pd.read_csv('PLGA_nanoparticles_dataset.csv')
        return model_size, model_ee, df, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, False

model_size, model_ee, df, models_loaded = load_models()

if not models_loaded:
    st.stop()

# ========== SIDEBAR - INPUT PARAMETERS ==========
st.sidebar.header("📋 Drug Properties")
st.sidebar.markdown("*Enter molecular descriptors for your drug*")

# Create two columns in sidebar for compact layout
col1, col2 = st.sidebar.columns(2)

with col1:
    mol_MW = st.number_input(
        "Mol. Weight (g/mol)",
        min_value=100.0,
        max_value=1000.0,
        value=420.5,
        step=10.0,
        help="Molecular weight of the drug"
    )
    
    mol_logP = st.number_input(
        "LogP",
        min_value=-5.0,
        max_value=10.0,
        value=3.1,
        step=0.1,
        help="Lipophilicity (octanol-water partition coefficient)"
    )
    
    mol_TPSA = st.number_input(
        "TPSA (Ų)",
        min_value=0.0,
        max_value=300.0,
        value=72.0,
        step=5.0,
        help="Topological Polar Surface Area"
    )
    
    mol_melting_point = st.number_input(
        "Melting Point (°C)",
        min_value=0.0,
        max_value=500.0,
        value=175.0,
        step=5.0,
        help="Melting point of the drug"
    )

with col2:
    mol_Hacceptors = st.number_input(
        "H-Acceptors",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        help="Number of hydrogen bond acceptors"
    )
    
    mol_Hdonors = st.number_input(
        "H-Donors",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        help="Number of hydrogen bond donors"
    )
    
    mol_heteroatoms = st.number_input(
        "Heteroatoms",
        min_value=0,
        max_value=30,
        value=7,
        step=1,
        help="Number of heteroatoms (N, O, S, etc.)"
    )
# After drug properties section in sidebar

st.sidebar.markdown("---")
st.sidebar.header("🔬 Polymer Settings")

# Radio button for mode selection
polymer_mode = st.sidebar.radio(
    "How do you want to specify PLGA?",
    ["Auto-optimize (Recommended)", "I have specific PLGA in my lab"],
    help="Auto-optimize: Model will find the best PLGA type for you\nSpecific PLGA: You specify which PLGA you have available"
)

if polymer_mode == "I have specific PLGA in my lab":
    st.sidebar.info("💡 Specify your available PLGA. Model will optimize other parameters.")
    
    # Get available PLGA options from dataset
    available_mw = sorted(df['polymer_MW'].unique())
    available_laga = sorted(df['LA/GA'].unique())
    
    user_polymer_mw = st.sidebar.selectbox(
        "PLGA Molecular Weight (kDa)",
        available_mw,
        index=available_mw.index(12.0) if 12.0 in available_mw else 0,
        help="Select the PLGA molecular weight you have in your lab"
    )
    
    user_la_ga = st.sidebar.selectbox(
        "LA/GA Ratio",
        available_laga,
        index=available_laga.index(3.0) if 3.0 in available_laga else 0,
        help="Lactide/Glycolide ratio of your PLGA"
    )
    
    # Store constraints
    polymer_constraints = {
        'polymer_MW': user_polymer_mw,
        'LA/GA': user_la_ga
    }
    
else:
    # No constraints - full optimization
    polymer_constraints = None

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Optimization Settings")
# ... rest of your code


target_size = st.sidebar.slider(
    "Target Particle Size (nm)",
    min_value=50,
    max_value=250,
    value=180,
    step=10,
    help="Maximum acceptable particle size for brain delivery"
)

min_ee = st.sidebar.slider(
    "Minimum EE% Target",
    min_value=50,
    max_value=95,
    value=70,
    step=5,
    help="Minimum acceptable entrapment efficiency"
)

n_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="How many formulations to recommend"
)

# Add information box in sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    """
    💡 **How to get molecular properties:**
    - ChemDraw: Calculate → Properties
    - PubChem: Search drug name
    - SwissADME: Free online tool
    - RDKit: Python library
    """
)

# Optimize button
optimize_button = st.sidebar.button(
    "🎯 OPTIMIZE FORMULATION",
    use_container_width=True,
    help="Click to generate optimized formulation recommendations"
)

# ========== OPTIMIZATION FUNCTION ==========
def optimize_formulation(drug_properties, target_size, min_ee, n_top):
    """Generate optimized formulation recommendations"""
    
    # Filter successful formulations from dataset
    successful = df[df['particle_size'] < target_size]
    
    if len(successful) < 20:
        st.warning(f"Only {len(successful)} formulations achieved <{target_size}nm. Relaxing constraint...")
        successful = df[df['particle_size'] < 250]
    
    # Extract parameter distributions
    formulation_space = {
        'polymer_MW': successful['polymer_MW'].unique().tolist(),
        'LA/GA': successful['LA/GA'].unique().tolist(),
        'drug/polymer': np.linspace(
            successful['drug/polymer'].quantile(0.1),
            successful['drug/polymer'].quantile(0.9),
            40
        ).tolist(),
        'surfactant_concentration': np.linspace(
            successful['surfactant_concentration'].min(),
            successful['surfactant_concentration'].quantile(0.9),
            25
        ).tolist(),
        'surfactant_HLB': successful['surfactant_HLB'].unique().tolist(),
        'aqueous/organic': successful['aqueous/organic'].unique().tolist(),
        'pH': successful['pH'].unique().tolist(),
        'solvent_polarity_index': successful['solvent_polarity_index'].unique().tolist()
    }
    
    # Generate candidates
    import random
    random.seed(42)
    np.random.seed(42)
    
    n_candidates = 20000
    candidates = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_candidates):
        if i % 1000 == 0:
            progress_bar.progress(i / n_candidates)
            status_text.text(f'Generating candidates... {i:,}/{n_candidates:,}')
        
        candidate = {
            **drug_properties,
            'polymer_MW': random.choice(formulation_space['polymer_MW']),
            'LA/GA': random.choice(formulation_space['LA/GA']),
            'drug/polymer': random.choice(formulation_space['drug/polymer']),
            'surfactant_concentration': random.choice(formulation_space['surfactant_concentration']),
            'surfactant_HLB': random.choice(formulation_space['surfactant_HLB']),
            'aqueous/organic': random.choice(formulation_space['aqueous/organic']),
            'pH': random.choice(formulation_space['pH']),
            'solvent_polarity_index': random.choice(formulation_space['solvent_polarity_index'])
        }
        candidates.append(candidate)
    
    progress_bar.progress(1.0)
    status_text.text('Predicting outcomes...')
    
    candidates_df = pd.DataFrame(candidates)
    
    # Predict
    features = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 
                'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms',
                'drug/polymer', 'surfactant_concentration', 'surfactant_HLB',
                'aqueous/organic', 'pH', 'solvent_polarity_index']
    
    X_cand = candidates_df[features]
    candidates_df['pred_size'] = model_size.predict(X_cand)
    candidates_df['pred_EE'] = model_ee.predict(X_cand)
    
    # Scoring
    candidates_df['size_score'] = np.where(
        candidates_df['pred_size'] <= target_size,
        (target_size - candidates_df['pred_size']) / target_size,
        -0.5
    )
    candidates_df['ee_score'] = (candidates_df['pred_EE'] - 50) / 50
    candidates_df['total_score'] = 0.7 * candidates_df['size_score'] + 0.3 * candidates_df['ee_score']
    
    # Get diverse top candidates
    top_candidates = candidates_df.nlargest(n_top * 3, 'total_score')
    
    diverse_top = []
    used_ratios = []
    
    for _, row in top_candidates.iterrows():
        ratio = row['drug/polymer']
        if not any(abs(ratio - used) < 0.02 for used in used_ratios):
            diverse_top.append(row)
            used_ratios.append(ratio)
        if len(diverse_top) >= n_top:
            break
    
    if len(diverse_top) < n_top:
        diverse_top = [row for _, row in top_candidates.head(n_top).iterrows()]
    
    recommendations = pd.DataFrame(diverse_top).sort_values('pred_size').reset_index(drop=True)
    
    progress_bar.empty()
    status_text.empty()
    
    # Statistics for display
    stats = {
        'n_candidates': n_candidates,
        'n_below_target': (candidates_df['pred_size'] < target_size).sum(),
        'n_below_150': (candidates_df['pred_size'] < 150).sum(),
        'n_above_ee': (candidates_df['pred_EE'] > min_ee).sum()
    }
    
    return recommendations, stats

# ========== MAIN CONTENT AREA ==========

    if optimize_button:
    drug_properties = {
        'mol_MW': mol_MW,
        'mol_logP': mol_logP,
        'mol_TPSA': mol_TPSA,
        'mol_melting_point': mol_melting_point,
        'mol_Hacceptors': mol_Hacceptors,
        'mol_Hdonors': mol_Hdonors,
        'mol_heteroatoms': mol_heteroatoms
    }
    
    # Show what mode is being used
    if polymer_constraints:
        st.info(f"🔒 **Using your PLGA:** MW = {polymer_constraints['polymer_MW']} kDa, LA/GA = {polymer_constraints['LA/GA']}")
        st.info("✨ **Optimizing:** Drug/polymer ratio, surfactant type & concentration, pH, aqueous/organic ratio, solvent polarity")
    else:
        st.info("🚀 **Full Auto-Optimization Mode** - Finding the best PLGA type and all formulation parameters")
    
    # Run optimization
    with st.spinner('🔬 Optimizing formulation parameters...'):
        recommendations, stats = optimize_formulation(
            drug_properties, 
            target_size, 
            min_ee, 
            n_recommendations,
            constraints=polymer_constraints  # Pass polymer constraints
        )
    
    # ... rest of display code
    # Run optimization
    
    # Success message
    st.success("✅ **Optimization Complete!** Found optimal formulation parameters.")
    
    # Display metrics
    st.markdown("### 📊 Optimization Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Candidates Generated",
            f"{stats['n_candidates']:,}"
        )
    
    with col2:
        st.metric(
            f"Predicted <{target_size}nm",
            f"{stats['n_below_target']:,}",
            f"{stats['n_below_target']/stats['n_candidates']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Best Size",
            f"{recommendations['pred_size'].min():.1f} nm"
        )
    
    with col4:
        st.metric(
            "Best EE",
            f"{recommendations['pred_EE'].max():.1f}%"
        )
    
    # Display recommendations table
    st.markdown("### 🎯 Recommended Formulations")
    
    # Format table for display
    display_df = recommendations.copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df = display_df[['Rank', 'drug/polymer', 'polymer_MW', 'LA/GA', 
                            'surfactant_concentration', 'surfactant_HLB',
                            'aqueous/organic', 'pH', 'pred_size', 'pred_EE']]
    
    display_df.columns = ['Rank', 'Drug/Polymer', 'Polymer MW (kDa)', 'LA/GA', 
                          'Surf. Conc (%)', 'Surf. HLB', 'Aq/Org', 'pH',
                          'Pred. Size (nm)', 'Pred. EE (%)']
    
    # Style the dataframe
    st.dataframe(
        display_df.style.format({
            'Drug/Polymer': '{:.4f}',
            'Polymer MW (kDa)': '{:.1f}',
            'LA/GA': '{:.2f}',
            'Surf. Conc (%)': '{:.2f}',
            'Surf. HLB': '{:.1f}',
            'Aq/Org': '{:.2f}',
            'pH': '{:.0f}',
            'Pred. Size (nm)': '{:.1f}',
            'Pred. EE (%)': '{:.1f}'
        }).background_gradient(subset=['Pred. Size (nm)'], cmap='RdYlGn_r')
          .background_gradient(subset=['Pred. EE (%)'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Visualizations
    st.markdown("### 📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart - Particle sizes
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.viridis(recommendations['pred_EE'] / 100)
        bars = ax1.barh(range(len(recommendations)), recommendations['pred_size'],
                       color=colors, edgecolor='black', linewidth=1.5)
        
        ax1.axvline(x=target_size, color='red', linestyle='--', 
                   linewidth=2, label=f'Target ({target_size} nm)')
        ax1.axvline(x=150, color='orange', linestyle=':', 
                   linewidth=2, label='Ideal (<150 nm)')
        
        ax1.set_yticks(range(len(recommendations)))
        ax1.set_yticklabels([f"Formulation {i+1}" for i in range(len(recommendations))])
        ax1.set_xlabel('Predicted Particle Size (nm)', fontweight='bold')
        ax1.set_title('Predicted Particle Sizes\n(color intensity = EE%)', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig1)
    
    with col2:
        # Scatter plot - Size vs EE
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        scatter = ax2.scatter(recommendations['pred_size'], recommendations['pred_EE'],
                            s=400, c=recommendations['total_score'], cmap='plasma',
                            edgecolors='black', linewidth=2, alpha=0.8)
        
        ax2.axvline(x=target_size, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=min_ee, color='blue', linestyle='--', alpha=0.5)
        
        for i in range(len(recommendations)):
            ax2.annotate(f"{i+1}", 
                        (recommendations.iloc[i]['pred_size'], recommendations.iloc[i]['pred_EE']),
                        fontsize=12, fontweight='bold', ha='center', va='center', color='white')
        
        ax2.set_xlabel('Predicted Particle Size (nm)', fontweight='bold')
        ax2.set_ylabel('Predicted EE%', fontweight='bold')
        ax2.set_title('Size vs EE Trade-off\n(color = overall score)', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.colorbar(scatter, ax=ax2, label='Overall Score')
        
        st.pyplot(fig2)
    
    # Lab Protocol
    st.markdown("### 🧪 Suggested Lab Protocol")
    
    best = recommendations.iloc[0]
    
    protocol_text = f"""
    **Starting with Formulation #1 (Best Overall Score)**
    
    **📋 Materials Required:**
    - Drug: Based on 1:{1/best['drug/polymer']:.0f} drug/polymer ratio
    - PLGA: MW = {best['polymer_MW']:.1f} kDa, LA/GA = {best['LA/GA']:.2f}
    - Surfactant: HLB {best['surfactant_HLB']:.1f}, Concentration {best['surfactant_concentration']:.2f}%
    - Aqueous/Organic Phase Ratio: {best['aqueous/organic']:.2f}:1
    - pH Adjustment: Encoded value {best['pH']:.0f}
    
    **🎯 Expected Outcomes:**
    - Particle Size: {best['pred_size']:.1f} ± 30 nm
    - Entrapment Efficiency: {best['pred_EE']:.1f} ± 10%
    
    **💡 Experimental Strategy:**
    1. Start with Formulation #1
    2. If particle size deviates >50 nm from prediction, try Formulation #2
    3. Test at least top 3 formulations to find best actual performer
    4. Use actual results to validate/refine predictions
    
    **⚠️ Notes:**
    - Predictions based on ML model (R² = 0.88 for size, 0.47 for EE)
    - Actual results may vary based on equipment and technique
    - Consider process parameters: stirring speed, temperature, sonication time
    """
    
    st.info(protocol_text)
    
    # Download options
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"formulation_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Generate PDF report (simplified text version)
        report = f"""
PLGA NANOPARTICLE FORMULATION OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DRUG PROPERTIES:
- Molecular Weight: {mol_MW} g/mol
- LogP: {mol_logP}
- TPSA: {mol_TPSA} Ų
- Melting Point: {mol_melting_point} °C
- H-Acceptors: {mol_Hacceptors}
- H-Donors: {mol_Hdonors}
- Heteroatoms: {mol_heteroatoms}

OPTIMIZATION PARAMETERS:
- Target Particle Size: <{target_size} nm
- Minimum EE: >{min_ee}%
- Recommendations Generated: {n_recommendations}

TOP RECOMMENDATION:
- Drug/Polymer Ratio: {best['drug/polymer']:.4f}
- Polymer MW: {best['polymer_MW']:.1f} kDa
- LA/GA: {best['LA/GA']:.2f}
- Surfactant HLB: {best['surfactant_HLB']:.1f}
- Surfactant Concentration: {best['surfactant_concentration']:.2f}%
- Predicted Size: {best['pred_size']:.1f} nm
- Predicted EE: {best['pred_EE']:.1f}%

ALL RECOMMENDATIONS:
{display_df.to_string()}
        """
        
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report,
            file_name=f"formulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Store results in session state
    st.session_state['last_recommendations'] = recommendations
    st.session_state['last_drug_properties'] = drug_properties

else:
    # Default view when no optimization has been run
    st.markdown("## 👋 Welcome to the PLGA Nanoformulation Optimizer")
    
    st.markdown("""
    This tool uses machine learning to predict optimal PLGA nanoparticle formulation parameters
    for brain-targeted drug delivery.
    
    ### 🎯 How to Use:
    1. **Enter drug properties** in the sidebar (MW, logP, TPSA, etc.)
    2. **Set optimization targets** (desired particle size and EE%)
    3. **Click "Optimize Formulation"** button
    4. **Review recommendations** and download results
    
    ### 📊 Model Performance:
    - **Particle Size Prediction:** R² = 0.88, MAE = ±22 nm
    - **EE% Prediction:** R² = 0.47, MAE = ±10%
    - **Training Data:** 433 PLGA formulations from literature
    
    ### 🧪 Example Use Case:
    For a CNS drug with MW=420 Da and logP=3.1, the optimizer typically recommends
    formulations achieving:
    - Particle size: 95-110 nm
    - Entrapment efficiency: 75-85%
    - Time saved: ~80% reduction in trial formulations
    """)
    
    # Display example results if available
    if 'last_recommendations' in st.session_state:
        st.markdown("### 📌 Last Optimization Results")
        st.dataframe(st.session_state['last_recommendations'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    Developed by Hardik Sood | IIT BHU Pharmaceutical Engineering | 
    Supervised by Dr. Ruchi Chawla
</div>

""", unsafe_allow_html=True)
