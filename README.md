# 🧬 PLGA Nanoparticle Formulation Optimizer

AI-powered tool for optimizing PLGA nanoparticle formulations for brain-targeted drug delivery.

**Developed by:** Hardik Sood  
**Institution:** IIT BHU, Department of Pharmaceutical Engineering  
**Supervisor:** Dr. Ruchi Chawla  

---

## 🌐 Live Application

**Access the app here:** https://plga-formulation-optimizer.streamlit.app/

---

## 📊 Model Performance

- **Particle Size Prediction:** R² = 0.88, MAE = ±22 nm
- **Entrapment Efficiency Prediction:** R² = 0.47, MAE = ±10%
- **Training Data:** 433 PLGA formulations from published literature (2024)

---

## 🎯 Features

✅ Predicts optimal formulation parameters for any drug  
✅ Generates 20,000 candidates and recommends top 5  
✅ Targets particle size <200 nm for BBB penetration  
✅ Maximizes entrapment efficiency  
✅ Provides complete lab protocols  
✅ Downloadable results (CSV + TXT)  
✅ Interactive visualizations  

---

## 🧪 How to Use

1. **Enter drug molecular properties:**
   - Molecular Weight (g/mol)
   - LogP (Lipophilicity)
   - TPSA (Topological Polar Surface Area)
   - Melting Point (°C)
   - H-Bond Acceptors/Donors
   - Heteroatoms

2. **Set optimization targets:**
   - Target particle size (default: <180 nm)
   - Minimum EE% (default: >70%)

3. **Click "Optimize Formulation"**

4. **Review recommendations:**
   - Top 5 formulations ranked by overall score
   - Predicted particle size and EE%
   - Complete formulation parameters
   - Lab preparation protocol

5. **Download results** as CSV or text report

---

## 💡 Example Results

For a CNS drug with:
- MW = 420 Da
- logP = 3.1
- TPSA = 72 Ų

**Recommended formulations achieve:**
- Particle size: 96-108 nm
- Entrapment efficiency: 77-83%
- Optimal drug/polymer ratio: 1:12 to 1:25

**Lab impact:** 70-80% reduction in trial experiments

---

## 📚 Technical Details

**Machine Learning:**
- Algorithm: XGBoost (Gradient Boosting)
- Features: 15 input parameters
- Targets: Particle size (nm), EE (%)

**Technologies:**
- Python 3.10
- Streamlit (web interface)
- Scikit-learn, XGBoost (ML)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)

---

## 📧 Contact

**Hardik Sood**  
B.Tech Pharmaceutical Engineering  
Indian Institute of Technology (BHU) Varanasi  
Email: hardik.sood.phe24@itbhu.ac.in


---

## 🙏 Acknowledgments

- Dr. Ruchi Chawla (Project Supervisor)
- Department of Pharmaceutical Engineering, IIT BHU
- Dataset source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12246394/#Sec6
