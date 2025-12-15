# Framingham Heart Study - Cardiovascular Disease Risk Analysis

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Peiprjs/MAI3002_project/HEAD?urlpath=%2Fdoc%2Ftree%2FCode.ipynb) [![DOI](https://zenodo.org/badge/1083256166.svg)](https://doi.org/10.5281/zenodo.17945400) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://framingham-heart.streamlit.app)

## Overview

This project provides an analysis of the Framingham Heart Study dataset to attempt to predict cardiovascular disease (CVD) risk using the following machine learning techniques: Logistic Regression, Decision Tree, Random Forest. 

## Methodology

### Data Preprocessing Pipeline
1. **Train-Test Split**: 80/20 split with stratification
2. **Missing Data Handling**: Three-tier imputation approach
3. **Outlier Detection**: IQR method (outliers retained for clinical validity)
4. **Skewness Correction**: PowerTransformer for features with |skewness| > 0.5
5. **Leakage Variable Removal**: Drop variables that leak information about the outcome
6. **Feature Selection**: RFE with multiple random seeds for robustness
7. **Correlation Filtering**: Remove highly correlated features (>0.90)

### Model Training
- **Stratified Sampling**: Ensures balanced representation of CVD cases
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Hyperparameter Tuning**: Interactive sliders for Random Forest optimization
- **Class Balancing**: Handles imbalanced dataset through class weights

## Technologies Used

- **Python**: Core programming language
- **Jupyter Notebook**: Interactive development environment
- **Streamlit**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning algorithms and preprocessing tools
- **Plotly & Seaborn**: Data visualization
- **SHAP**: Model interpretability
- **Matplotlib**: Static visualizations

## Citation

If you use this project in your research, please cite:

```bibtex
@software{framingham_cvd_analysis,
  title = {Framingham Heart Study - Cardiovascular Disease Risk Analysis},
  author = {Jakub Záboj, Mar Roca},
  year = {2025},
  doi = {10.5281/zenodo.17462955},
  url = {https://github.com/peipr-access/MAI3002-Framingham_Heart_Study}
}
```

> Good luck, little code. I therefore sail ship to this code, never to be touched again. I hope your voyage across the sea of demoing is calm and deovoid of bugs and other integration hell beasts.
