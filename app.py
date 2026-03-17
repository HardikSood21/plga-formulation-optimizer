"""
Multi-Polymer Nanoparticle Formulation Optimizer
PLGA + Chitosan Models
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
import random

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="NanoFormula AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        color: #2E86AB;
        text-align: center;
        padding: 15px 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 16px;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    .polymer-badge-plga {
        background-color: #2E86AB;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .polymer-badge-chitosan {
        background-color: #27AE60;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown('<div class="main-header">🧬 NanoFormula AI</div>',  
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Polymer Nanoparticle Formulation Optimizer<br>Dr. Ruchi Chawla\'s Lab | IIT BHU Pharmaceutical Engineering</div>', 
            unsafe_allow_html=True)

# ========== LOAD ALL MODELS ==========
@st.cache_resource
def load_all_models():
    """Load all pre-trained models and datasets"""
    models = {}
    
    # PLGA models
    try:
        with open('model_particle_size_final.pkl', 'rb') as f:
            models['plga_size'] = pickle.load(f)
        with open('model_ee_final.pkl', 'rb') as f:
            models['plga_ee'] = pickle.load(f)
        models['plga_data'] = pd.read_csv('PLGA_nanoparticles_dataset.csv')
        models['plga_loaded'] = True
    except:
        models['plga_loaded'] = False
    
    # Chitosan model
    try:
        with open('model_chitosan_size.pkl', 'rb') as f:
            models['chitosan_size'] = pickle.load(f)
        with open('chitosan_features.pkl', 'rb') as f:
            models['chitosan_features'] = pickle.load(f)
        models['chitosan_data'] = pd.read_csv('chitosan_nanoparticles_dataset.csv')
        models['chitosan_loaded'] = True
    except:
        models['chitosan_loaded'] = False
    
    return models

models = load_all_models()

# ========== POLYMER SELECTION ==========
st.sidebar.header("🔬 Select Polymer System")

polymer_options = []
if models.get('plga_loaded'):
    polymer_options.append("PLGA (Advanced - 433 formulations)")
if models.get('chitosan_loaded'):
    polymer_options.append("Chitosan (44 formulations)")

if not polymer_options:
    st.error("No models loaded! Check your files.")
    st.stop()

selected_polymer = st.sidebar.radio(
    "Choose polymer type:",
    polymer_options,
    help="PLGA: Full model with drug properties\nChitosan: Predicts particle size from polymer parameters"
)

st.sidebar.markdown("---")

# ========== PLGA MODE ==========
if selected_polymer.startswith("PLGA"):
    
    st.markdown("### 💊 PLGA Nanoparticle Optimizer")
    st.info("**Model:** R² = 0.88 | **Training data:** 433 formulations | **Predicts:** Particle Size + EE%")
    
    # Sidebar inputs
    st.sidebar.header("📋 Drug Properties")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        mol_MW = st.number_input("Mol. Weight (g/mol)", 100.0, 1000.0, 420.5, 10.0)
        mol_logP = st.number_input("LogP", -5.0, 10.0, 3.1, 0.1)
        mol_TPSA = st.number_input("TPSA (Ų)", 0.0, 300.0, 72.0, 5.0)
        mol_mp = st.number_input("Melting Point (°C)", 0.0, 500.0, 175.0, 5.0)
    with col2:
        mol_Hacc = st.number_input("H-Acceptors", 0, 20, 5, 1)
        mol_Hdon = st.number_input("H-Donors", 0, 10, 2, 1)
        mol_het = st.number_input("Heteroatoms", 0, 30, 7, 1)
    
    # Polymer settings
    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Polymer Settings")
    
    polymer_mode = st.sidebar.radio(
        "PLGA Selection:",
        ["Auto-optimize", "I have specific PLGA"]
    )
    
    if polymer_mode == "I have specific PLGA":
        df_plga = models['plga_data']
        available_mw = sorted(df_plga['polymer_MW'].unique())
        available_laga = sorted(df_plga['LA/GA'].unique())
        
        user_polymer_mw = st.sidebar.selectbox("PLGA MW (kDa)", available_mw)
        user_la_ga = st.sidebar.selectbox("LA/GA Ratio", available_laga)
        polymer_constraints = {'polymer_MW': user_polymer_mw, 'LA/GA': user_la_ga}
    else:
        polymer_constraints = None
    
    # Optimization settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Targets")
    target_size = st.sidebar.slider("Target Size (nm)", 50, 250, 180, 10)
    min_ee = st.sidebar.slider("Min EE%", 50, 95, 70, 5)
    n_recs = st.sidebar.slider("# Recommendations", 3, 10, 5, 1)
    
    # Optimize button
    optimize = st.sidebar.button("🎯 OPTIMIZE PLGA", use_container_width=True)
    
    if optimize:
        drug_properties = {
            'mol_MW': mol_MW, 'mol_logP': mol_logP, 'mol_TPSA': mol_TPSA,
            'mol_melting_point': mol_mp, 'mol_Hacceptors': mol_Hacc,
            'mol_Hdonors': mol_Hdon, 'mol_heteroatoms': mol_het
        }
        
        with st.spinner('🔬 Optimizing PLGA formulation...'):
            df_plga = models['plga_data']
            successful = df_plga[df_plga['particle_size'] < target_size]
            
            if len(successful) < 20:
                successful = df_plga[df_plga['particle_size'] < 250]
            
            formulation_space = {
                'polymer_MW': successful['polymer_MW'].unique().tolist(),
                'LA/GA': successful['LA/GA'].unique().tolist(),
                'drug/polymer': np.linspace(
                    successful['drug/polymer'].quantile(0.1),
                    successful['drug/polymer'].quantile(0.9), 40).tolist(),
                'surfactant_concentration': np.linspace(
                    successful['surfactant_concentration'].min(),
                    successful['surfactant_concentration'].quantile(0.9), 25).tolist(),
                'surfactant_HLB': successful['surfactant_HLB'].unique().tolist(),
                'aqueous/organic': successful['aqueous/organic'].unique().tolist(),
                'pH': successful['pH'].unique().tolist(),
                'solvent_polarity_index': successful['solvent_polarity_index'].unique().tolist()
            }
            
            if polymer_constraints:
                if 'polymer_MW' in polymer_constraints:
                    formulation_space['polymer_MW'] = [polymer_constraints['polymer_MW']]
                if 'LA/GA' in polymer_constraints:
                    formulation_space['LA/GA'] = [polymer_constraints['LA/GA']]
            
            random.seed(42)
            np.random.seed(42)
            
            candidates = []
            for _ in range(20000):
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
            
            cand_df = pd.DataFrame(candidates)
            
            plga_features = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA',
                           'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms',
                           'drug/polymer', 'surfactant_concentration', 'surfactant_HLB',
                           'aqueous/organic', 'pH', 'solvent_polarity_index']
            
            X_cand = cand_df[plga_features]
            cand_df['pred_size'] = models['plga_size'].predict(X_cand)
            cand_df['pred_EE'] = models['plga_ee'].predict(X_cand)
            
            cand_df['size_score'] = np.where(
                cand_df['pred_size'] <= target_size,
                (target_size - cand_df['pred_size']) / target_size, -0.5)
            cand_df['ee_score'] = (cand_df['pred_EE'] - 50) / 50
            cand_df['total_score'] = 0.7 * cand_df['size_score'] + 0.3 * cand_df['ee_score']
            
            top = cand_df.nlargest(n_recs * 3, 'total_score')
            diverse = []
            used_ratios = []
            for _, row in top.iterrows():
                ratio = row['drug/polymer']
                if not any(abs(ratio - u) < 0.02 for u in used_ratios):
                    diverse.append(row)
                    used_ratios.append(ratio)
                if len(diverse) >= n_recs:
                    break
            if len(diverse) < n_recs:
                diverse = [row for _, row in top.head(n_recs).iterrows()]
            
            recs = pd.DataFrame(diverse).sort_values('pred_size').reset_index(drop=True)
        
        # Display results
        st.success("✅ PLGA Optimization Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates", "20,000")
        col2.metric(f"< {target_size}nm", f"{(cand_df['pred_size'] < target_size).sum():,}")
        col3.metric("Best Size", f"{recs['pred_size'].min():.1f} nm")
        col4.metric("Best EE", f"{recs['pred_EE'].max():.1f}%")
        
        st.markdown("### 🎯 Recommended Formulations")
        
        display_df = recs[['drug/polymer', 'polymer_MW', 'LA/GA',
                          'surfactant_concentration', 'surfactant_HLB',
                          'aqueous/organic', 'pH', 'pred_size', 'pred_EE']].copy()
        display_df.columns = ['Drug/Polymer', 'PLGA MW (kDa)', 'LA/GA',
                             'Surf. Conc (%)', 'Surf. HLB', 'Aq/Org', 'pH',
                             'Pred. Size (nm)', 'Pred. EE (%)']
        
        st.dataframe(display_df.style.format({
            'Drug/Polymer': '{:.4f}', 'PLGA MW (kDa)': '{:.1f}',
            'LA/GA': '{:.2f}', 'Surf. Conc (%)': '{:.2f}',
            'Surf. HLB': '{:.1f}', 'Pred. Size (nm)': '{:.1f}',
            'Pred. EE (%)': '{:.1f}'
        }).background_gradient(subset=['Pred. Size (nm)'], cmap='RdYlGn_r')
          .background_gradient(subset=['Pred. EE (%)'], cmap='RdYlGn'),
            use_container_width=True)
        
        # Plots
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            colors = plt.cm.viridis(recs['pred_EE'] / 100)
            ax1.barh(range(len(recs)), recs['pred_size'], color=colors, 
                    edgecolor='black', linewidth=1.5)
            ax1.axvline(x=target_size, color='red', linestyle='--', linewidth=2,
                       label=f'Target ({target_size} nm)')
            ax1.set_yticks(range(len(recs)))
            ax1.set_yticklabels([f"Form {i+1}" for i in range(len(recs))])
            ax1.set_xlabel('Predicted Size (nm)', fontweight='bold')
            ax1.set_title('Predicted Particle Sizes', fontweight='bold')
            ax1.legend()
            ax1.grid(axis='x', alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            scatter = ax2.scatter(recs['pred_size'], recs['pred_EE'],
                                s=400, c=recs['total_score'], cmap='plasma',
                                edgecolors='black', linewidth=2)
            ax2.axvline(x=target_size, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=min_ee, color='blue', linestyle='--', alpha=0.5)
            for i in range(len(recs)):
                ax2.annotate(f"{i+1}", (recs.iloc[i]['pred_size'], recs.iloc[i]['pred_EE']),
                           fontsize=12, fontweight='bold', ha='center', va='center', color='white')
            ax2.set_xlabel('Predicted Size (nm)', fontweight='bold')
            ax2.set_ylabel('Predicted EE%', fontweight='bold')
            ax2.set_title('Size vs EE Trade-off', fontweight='bold')
            ax2.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='Score')
            st.pyplot(fig2)
        
        # Download
        csv = display_df.to_csv(index=False)
        st.download_button("📥 Download Results (CSV)", csv,
                          f"plga_recommendations_{datetime.now().strftime('%Y%m%d')}.csv")

# ========== CHITOSAN MODE ==========
elif selected_polymer.startswith("Chitosan"):
    
    st.markdown("### 🧪 Chitosan Nanoparticle Optimizer")
    st.info("**Model:** R² = 0.83 | **Training data:** 44 formulations | **Predicts:** Particle Size")
    
    st.warning("""
    ⚠️ **Limitations of Chitosan Model:**
    - Predicts particle size only (no EE% - data not available)
    - No drug-specific predictions (blank nanoparticles)
    - 44 training samples (vs 433 for PLGA)
    - Based on 4 chitosan types: HMW (310 kDa), LMW (50 kDa), 20 kDa, 5 kDa
    """)
    
    # Sidebar inputs
    st.sidebar.header("🧪 Chitosan Parameters")
    
    chitosan_mode = st.sidebar.radio(
        "Chitosan Selection:",
        ["Auto-optimize (find best type)", "I have specific chitosan"]
    )
    
    if chitosan_mode == "I have specific chitosan":
        chitosan_mw = st.sidebar.selectbox(
            "Chitosan MW (kDa)",
            [5, 20, 50, 310],
            format_func=lambda x: f"{x} kDa ({'HMW' if x==310 else 'LMW' if x==50 else f'{x} kDa'})"
        )
        fix_mw = True
    else:
        chitosan_mw = None
        fix_mw = False
    
    # Optimization settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Targets")
    target_size_cs = st.sidebar.slider("Target Size (nm)", 50, 250, 150, 10, key="cs_size")
    n_recs_cs = st.sidebar.slider("# Recommendations", 3, 10, 5, 1, key="cs_recs")
    
    # Optimize button
    optimize_cs = st.sidebar.button("🎯 OPTIMIZE CHITOSAN", use_container_width=True)
    
    if optimize_cs:
        with st.spinner('🔬 Optimizing chitosan formulation...'):
            
            chitosan_features = models['chitosan_features']
            
            # Search space
            if fix_mw:
                mw_options = [chitosan_mw]
            else:
                mw_options = [5, 20, 50, 310]
            
            conc_options = np.linspace(0.10, 1.0, 25).tolist()
            tpp_options = np.linspace(0.10, 1.0, 25).tolist()
            
            random.seed(42)
            np.random.seed(42)
            
            n_candidates = 10000
            candidates = []
            
            for _ in range(n_candidates):
                mw = random.choice(mw_options)
                conc = random.choice(conc_options)
                tpp = random.choice(tpp_options)
                
                candidate = {
                    'chitosan_MW': mw,
                    'chitosan_conc': conc,
                    'TPP_conc': tpp,
                    'chitosan_TPP_ratio': conc / tpp,
                    'conc_x_TPP': conc * tpp,
                    'MW_x_conc': mw * conc,
                    'MW_x_TPP': mw * tpp,
                    'log_MW': np.log10(mw),
                    'total_solute': conc + tpp,
                    'chitosan_fraction': conc / (conc + tpp)
                }
                candidates.append(candidate)
            
            cand_df = pd.DataFrame(candidates)
            X_cand = cand_df[chitosan_features]
            cand_df['pred_size'] = models['chitosan_size'].predict(X_cand)
            
            # Score
            cand_df['score'] = np.where(
                cand_df['pred_size'] <= target_size_cs,
                (target_size_cs - cand_df['pred_size']) / target_size_cs, -0.5)
            
            # Get diverse recommendations
            top = cand_df.nlargest(n_recs_cs * 3, 'score')
            diverse = []
            used_mw = set()
            
            for _, row in top.iterrows():
                mw = row['chitosan_MW']
                if mw not in used_mw or len(diverse) < 2:
                    diverse.append(row)
                    used_mw.add(mw)
                if len(diverse) >= n_recs_cs:
                    break
            
            if len(diverse) < n_recs_cs:
                diverse = [row for _, row in top.head(n_recs_cs).iterrows()]
            
            recs_cs = pd.DataFrame(diverse).sort_values('pred_size').reset_index(drop=True)
        
        # Display results
        st.success("✅ Chitosan Optimization Complete!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates", f"{n_candidates:,}")
        col2.metric(f"< {target_size_cs}nm", f"{(cand_df['pred_size'] < target_size_cs).sum():,}")
        col3.metric("Best Size", f"{recs_cs['pred_size'].min():.1f} nm")
        
        # Results table
        st.markdown("### 🎯 Recommended Formulations")
        
        mw_type_map = {5: '5 kDa', 20: '20 kDa', 50: 'LMW (~50 kDa)', 310: 'HMW (~310 kDa)'}
        
        display_cs = recs_cs[['chitosan_MW', 'chitosan_conc', 'TPP_conc',
                             'chitosan_TPP_ratio', 'pred_size']].copy()
        display_cs['Chitosan Type'] = display_cs['chitosan_MW'].map(mw_type_map)
        display_cs = display_cs[['Chitosan Type', 'chitosan_MW', 'chitosan_conc',
                                'TPP_conc', 'chitosan_TPP_ratio', 'pred_size']]
        display_cs.columns = ['Chitosan Type', 'MW (kDa)', 'Chitosan Conc (mg/mL)',
                             'TPP Conc (mg/mL)', 'CS:TPP Ratio', 'Pred. Size (nm)']
        
        st.dataframe(display_cs.style.format({
            'MW (kDa)': '{:.0f}',
            'Chitosan Conc (mg/mL)': '{:.2f}',
            'TPP Conc (mg/mL)': '{:.2f}',
            'CS:TPP Ratio': '{:.2f}',
            'Pred. Size (nm)': '{:.1f}'
        }).background_gradient(subset=['Pred. Size (nm)'], cmap='RdYlGn_r'),
            use_container_width=True)
        
        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            bar_colors = ['#27AE60' if s < target_size_cs else '#E74C3C' 
                         for s in recs_cs['pred_size']]
            ax1.barh(range(len(recs_cs)), recs_cs['pred_size'],
                    color=bar_colors, edgecolor='black', linewidth=1.5)
            ax1.axvline(x=target_size_cs, color='red', linestyle='--', linewidth=2,
                       label=f'Target ({target_size_cs} nm)')
            ax1.set_yticks(range(len(recs_cs)))
            ax1.set_yticklabels([f"Form {i+1}" for i in range(len(recs_cs))])
            ax1.set_xlabel('Predicted Size (nm)', fontweight='bold')
            ax1.set_title('Chitosan NP - Predicted Sizes', fontweight='bold')
            ax1.legend()
            ax1.grid(axis='x', alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            mw_colors = {5: '#F39C12', 20: '#3498DB', 50: '#2ECC71', 310: '#E74C3C'}
            for _, row in recs_cs.iterrows():
                color = mw_colors.get(row['chitosan_MW'], 'gray')
                ax2.scatter(row['chitosan_conc'], row['pred_size'],
                          s=300, c=color, edgecolors='black', linewidth=2, alpha=0.8,
                          label=mw_type_map.get(row['chitosan_MW'], ''))
            ax2.axhline(y=target_size_cs, color='red', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Chitosan Conc (mg/mL)', fontweight='bold')
            ax2.set_ylabel('Predicted Size (nm)', fontweight='bold')
            ax2.set_title('Chitosan Conc vs Predicted Size', fontweight='bold')
            handles, labels = ax2.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax2.legend(unique.values(), unique.keys())
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)
        
        # Lab Protocol
        st.markdown("### 🧪 Suggested Lab Protocol")
        best = recs_cs.iloc[0]
        st.info(f"""
        **Starting Formulation (Best Predicted Size: {best['pred_size']:.1f} nm)**
        
        📋 **Materials:**
        - Chitosan: {mw_type_map.get(best['chitosan_MW'], '')} ({best['chitosan_MW']:.0f} kDa)
        - Sodium Tripolyphosphate (TPP)
        - 1% Acetic acid solution
        - Deionized water
        
        📋 **Preparation:**
        1. Dissolve chitosan at {best['chitosan_conc']:.2f} mg/mL in 1% acetic acid
        2. Prepare TPP solution at {best['TPP_conc']:.2f} mg/mL in deionized water
        3. CS:TPP ratio = {best['chitosan_TPP_ratio']:.2f}
        4. Add TPP dropwise to chitosan under magnetic stirring (500-700 RPM)
        5. Stir for 30 minutes at room temperature
        6. Characterize by DLS for particle size, PDI, and zeta potential
        
        ⚠️ **Note:** Predictions based on model with R² = 0.83 (±15-20 nm accuracy)
        """)
        
        # Download
        csv_cs = display_cs.to_csv(index=False)
        st.download_button("📥 Download Results (CSV)", csv_cs,
                          f"chitosan_recommendations_{datetime.now().strftime('%Y%m%d')}.csv")

# ========== DEFAULT VIEW (No optimization run yet) ==========
if not (selected_polymer.startswith("PLGA") and 'optimize' in dir() and optimize) and \
   not (selected_polymer.startswith("Chitosan") and 'optimize_cs' in dir() and optimize_cs):
    
    if selected_polymer.startswith("PLGA"):
        st.markdown("### 👈 Enter drug properties and click 'Optimize PLGA'")
    else:
        st.markdown("### 👈 Set chitosan parameters and click 'Optimize Chitosan'")
    
    # Model comparison table
    st.markdown("### 📊 Available Models")
    
    comparison_data = {
        'Feature': ['Polymer', 'Training Samples', 'R² Score', 'MAE',
                    'Predicts Size?', 'Predicts EE?', 'Drug-Specific?', 'Status'],
        'PLGA Model': ['PLGA', '433', '0.88', '±22 nm',
                       '✅ Yes', '✅ Yes (R²=0.47)', '✅ Yes (65 drugs)', '🟢 Production'],
        'Chitosan Model': ['Chitosan-TPP', '44', '0.83', '±15 nm',
                          '✅ Yes', '❌ No data', '❌ Blank NPs only', '🟡 Basic']
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 13px;'>
    NanoFormula AI v2.0 | Developed by Hardik Sood | IIT BHU Pharmaceutical Engineering |  
    Supervised by Dr. Ruchi Chawla<br>
    PLGA Model: R² = 0.88 (433 samples) | Chitosan Model: R² = 0.83 (44 samples)
</div>
""", unsafe_allow_html=True)



