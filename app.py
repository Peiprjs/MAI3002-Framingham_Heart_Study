import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFE
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

from imputation_functions import drop_high_missing_cols, knn_impute, impute_simple_central
from functions import distplots

st.set_page_config(layout="wide", page_title="Framingham Heart Study")

warnings.filterwarnings('ignore')

# Load data
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    DatasetURL = "https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv"
    try:
        data = pd.read_csv(DatasetURL)
    except (Exception) as e:
        # Fallback to local file if URL fails
        try:
            data = pd.read_csv("Framingham Dataset.csv")
        except FileNotFoundError:
            st.error("Dataset not found. Please ensure the data file is available.")
            raise
    return data

@st.cache_data
def preprocess_data(data):
    """Preprocess data with imputation"""
    # Identify binary columns and time columns
    binary_cols = ['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'PREVCHD', 'PREVAP', 
                   'PREVMI', 'PREVSTRK', 'PREVHYP', 'ANYCHD', 'ANGINA', 
                   'HOSPMI', 'MI_FCHD', 'DEATH', 'STROKE', 'CVD', 'HYPERTEN']
    time_cols = [col for col in data.columns if col.startswith('TIME')]
    
    # Suppress imputation function output to avoid cluttering the UI
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Drop high missing columns
        data_dropped = drop_high_missing_cols(data, threshold=0.50)
        
        # KNN imputation
        data_knn = knn_impute(data_dropped, min_thresh=0.02, max_thresh=0.50, n_neighbors=5)
        
        # Simple imputation
        data_imputed = impute_simple_central(data_knn)
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    return data_imputed, binary_cols, time_cols

# Load data
data_raw = load_data()
data_imputed, binary_cols, time_cols = preprocess_data(data_raw)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title='Navigation',
        options=['Abstract', 'Exploratory Data Analysis', 'Statistical Analysis', 'Machine Learning Results', 'Conclusion'],
        menu_icon='heart-pulse',
        icons=['bookmark-check', 'bar-chart', 'calculator', 'cpu', 'check2-circle'],
        default_index=0,
    )

# Abstract Section
if selected == 'Abstract':
    st.title("Framingham Heart Study Analysis")
    st.markdown("""
    ## Abstract
        
    Navigate through the sections using the sidebar to explore the detailed analysis.
    """.format(data_raw.shape[0], data_raw.shape[1]))
    
    # Display basic statistics
    st.subheader("Dataset Preview")
    st.dataframe(data_raw.head(10))

# Data Overview Section
elif selected == 'Exploratory Data Analysis':
    st.title("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Participants", f"{data_raw.shape[0]:,}")
    with col2:
        st.metric("Total Features", data_raw.shape[1])

    data_type = st.segmented_control("Select data type", ["Raw Data", "Imputed Data"], default = "Imputed Data")
    data_to_use = data_raw if data_type == "Raw Data" else data_imputed

    st.subheader("Dataset Statistics")
    st.dataframe(data_to_use.describe())

    st.subheader("Data Distribution")

    # Select numeric columns for visualization
    numeric_cols = data_to_use.select_dtypes(include=['number']).columns.tolist()
    selected_col = st.selectbox("Select a variable to visualize", numeric_cols)

    # Create histogram

    # Create histogram with KDE
    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(x=data_to_use[selected_col], nbinsx=30, name='Histogram'))

    # Add KDE
    kde_x = np.linspace(data_to_use[selected_col].min(), data_to_use[selected_col].max(), 100)
    kde = stats.gaussian_kde(data_to_use[selected_col].dropna())
    kde_y = kde(kde_x)
    fig.add_trace(go.Scatter(x=kde_x, y=kde_y * len(data_to_use[selected_col]) * (
            data_to_use[selected_col].max() - data_to_use[selected_col].min()) / 30,
                             mode='lines', name='KDE', line=dict(color='red')))

    fig.update_layout(title=f'Distribution of {selected_col}',
                      xaxis_title=selected_col,
                      template='plotly_white',
                      showlegend=False,
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Analysis")
    all_vars = data_imputed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    key_vars = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'BMI', 'HEARTRATE', 'GLUCOSE', 'CVD']

    options = ["Key Variables", "All Variables"]
    var_type = st.segmented_control("Select variables to analyze", options, selection_mode="single", default = "Key Variables")
    selected_vars = key_vars if var_type == "Key Variables" else all_vars
    available_vars = [var for var in selected_vars if var in data_imputed.columns]

    if len(available_vars) > 1:
        corr_matrix = data_imputed[available_vars].corr()

        fig_corr = px.imshow(corr_matrix,
                             text_auto='.2f',
                             aspect="auto",
                             color_continuous_scale='RdBu_r',
                             title=f'Correlation Matrix of {var_type}')
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Pairplot Analysis")
        st.warning("Due to memory constraints, only Key Variables can be shown while in the deployed app.")

        options = ["Pretty", "Resource-friendly"]
        efficiency = st.segmented_control("How should the data be displayed?", options, selection_mode="single",
                                        default="Resource-friendly")
        if efficiency == "Pretty":
            st.warning("Will take a long time to run if on own hardware, won't run on deployed app.")
            @st.cache_resource(show_spinner=True, show_time=True)
            def pairplots(available_vars):
                fig = make_subplots(rows=len(available_vars), cols=len(available_vars),
                                    subplot_titles=[f"{v1} vs {v2}" for v1 in available_vars for v2 in available_vars])

                for i, var1 in enumerate(available_vars, 1):
                    for j, var2 in enumerate(available_vars, 1):
                        if var1 == var2:
                            # Histogram on diagonal
                            fig.add_trace(
                                go.Histogram(x=data_to_use[var1], name=var1),
                                row=i, col=j
                            )
                        else:
                            # Scatter plot off diagonal
                            fig.add_trace(
                                go.Scatter(x=data_to_use[var2], y=data_to_use[var1],
                                           mode='markers', marker=dict(size=3),
                                           name=f"{var1} vs {var2}"),
                                row=i, col=j
                            )

                fig.update_layout(height=200 * len(available_vars), showlegend=False)
                return fig
            st.plotly_chart(pairplots(key_vars), use_container_width=True)
        else:
            @st.cache_resource(show_spinner=True, show_time=True)
            def pairplots_eco(data):
                fig = sns.pairplot(data, hue="CVD")
                return fig
            selected_vars = key_vars
            available_vars = [var for var in selected_vars if var in data_imputed.columns]
            st.pyplot(pairplots_eco(data_to_use[available_vars]))


# Statistical Analysis Section
elif selected == 'Statistical Analysis':
    st.title("Statistical Analysis")
    
    st.markdown("""
    This section presents the statistical analysis results for the key research questions 
    about cardiovascular disease risk factors.
    """)
    
    # Research Question 1: Cholesterol and Smoking
    st.header("1. Total Cholesterol: Smokers vs Non-Smokers")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Calculate statistics
        mean_cholesterol = data_imputed.groupby('CURSMOKE')['TOTCHOL'].describe()
        st.dataframe(mean_cholesterol)
        
        # Perform t-test
        nonsmokers_chol = data_imputed[data_imputed['CURSMOKE'] == 0]['TOTCHOL']
        smokers_chol = data_imputed[data_imputed['CURSMOKE'] == 1]['TOTCHOL']
        levene_stat, levene_p = stats.levene(nonsmokers_chol, smokers_chol)
        
        t_stat, p_value = stats.ttest_ind(smokers_chol, nonsmokers_chol,
                                          equal_var=False if levene_p < 0.05 else True)
        
        st.metric("T-statistic", f"{t_stat:.4f}")
        st.metric("P-value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            st.success("The difference in mean total cholesterol is statistically significant.")
        else:
            st.info("No statistically significant difference in mean total cholesterol.")
    
    with col2:
        # Visualization
        fig = px.box(data_imputed, x='CURSMOKE', y='TOTCHOL',
                    labels={'CURSMOKE': 'Smoking Status', 'TOTCHOL': 'Total Cholesterol'},
                    title='Total Cholesterol by Smoking Status')
        fig.update_xaxes(ticktext=['Non-Smoker', 'Smoker'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    # Research Question 2: Blood Pressure and Smoking
    st.header("2. Blood Pressure: Smokers vs Non-Smokers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Systolic Blood Pressure")
        nonsmokers_sys = data_imputed[data_imputed['CURSMOKE'] == 0]['SYSBP']
        smokers_sys = data_imputed[data_imputed['CURSMOKE'] == 1]['SYSBP']
        
        t_stat_sys, p_value_sys = stats.ttest_ind(nonsmokers_sys, smokers_sys)
        
        fig_sys = px.box(data_imputed, x='CURSMOKE', y='SYSBP',
                        labels={'CURSMOKE': 'Smoking Status', 'SYSBP': 'Systolic BP (mmHg)'},
                        title='Systolic Blood Pressure by Smoking Status')
        fig_sys.update_xaxes(ticktext=['Non-Smoker', 'Smoker'], tickvals=[0, 1])
        st.plotly_chart(fig_sys, use_container_width=True)
        
        if p_value_sys < 0.05:
            st.success(f"Significant difference (p={p_value_sys:.4f})")
        else:
            st.info(f"No significant difference (p={p_value_sys:.4f})")
    
    with col2:
        st.subheader("Diastolic Blood Pressure")
        nonsmokers_dias = data_imputed[data_imputed['CURSMOKE'] == 0]['DIABP']
        smokers_dias = data_imputed[data_imputed['CURSMOKE'] == 1]['DIABP']
        
        t_stat_dias, p_value_dias = stats.ttest_ind(nonsmokers_dias, smokers_dias)
        
        fig_dias = px.box(data_imputed, x='CURSMOKE', y='DIABP',
                         labels={'CURSMOKE': 'Smoking Status', 'DIABP': 'Diastolic BP (mmHg)'},
                         title='Diastolic Blood Pressure by Smoking Status')
        fig_dias.update_xaxes(ticktext=['Non-Smoker', 'Smoker'], tickvals=[0, 1])
        st.plotly_chart(fig_dias, use_container_width=True)
        
        if p_value_dias < 0.05:
            st.success(f"Significant difference (p={p_value_dias:.4f})")
        else:
            st.info(f"No significant difference (p={p_value_dias:.4f})")
    
    # Research Question 3: Age and CVD in Smokers vs Non-Smokers
    st.header("3. Age Analysis: CVD Patients by Smoking Status")
    
    patients_with_cvd = data_imputed[data_imputed['CVD'] == 1].copy()
    
    if not patients_with_cvd.empty:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            age_stats = patients_with_cvd.groupby('CURSMOKE')['AGE'].describe()
            st.dataframe(age_stats)
            
            smokers_age_cvd = patients_with_cvd[patients_with_cvd['CURSMOKE'] == 1]['AGE']
            nonsmokers_age_cvd = patients_with_cvd[patients_with_cvd['CURSMOKE'] == 0]['AGE']
            
            t_stat_age, p_value_age = stats.ttest_ind(smokers_age_cvd, nonsmokers_age_cvd)
            
            st.metric("T-statistic", f"{t_stat_age:.4f}")
            st.metric("P-value", f"{p_value_age:.4f}")
            
            if p_value_age < 0.05:
                st.success("Significant age difference between smokers and non-smokers with CVD.")
            else:
                st.info("No significant age difference between smokers and non-smokers with CVD.")
        
        with col2:
            fig_age = px.box(patients_with_cvd, x='CURSMOKE', y='AGE',
                            labels={'CURSMOKE': 'Smoking Status', 'AGE': 'Age (years)'},
                            title='Age of CVD Patients by Smoking Status')
            fig_age.update_xaxes(ticktext=['Non-Smoker', 'Smoker'], tickvals=[0, 1])
            st.plotly_chart(fig_age, use_container_width=True)

# Machine Learning Results Section
elif selected == 'Machine Learning Results':
    st.title("Machine Learning Results")
    
    st.markdown("""
    This section presents the results of machine learning models trained to predict 
    cardiovascular disease outcomes using advanced preprocessing techniques.
    """)
    
    # Prepare data for ML with proper train-test split before imputation
    from processing_functions import train_test_imputation, apply_skewness_correction

    if 'CVD' in data_raw.columns:
        # First, split the raw data
        CategoryColumn = 'CVD'
        X = data_raw.drop(columns=[CategoryColumn])
        y = data_raw[CategoryColumn]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2025)
        
        with st.expander("Data Preprocessing Steps", expanded=False):
            st.markdown("""
            1. **Train-Test Split**: 80-20 split with stratification
            2. **Imputation**: Learn from training data and apply to both training and testing set
            3. **Skewness Correction**: PowerTransformer (Yeo-Johnson method)
            4. **Feature Selection**: Recursive Feature Elimination (RFE)
            """)
        
        # Apply train-test imputation
        with st.spinner("Applying imputation..."):
            X_train_imp, X_test_imp = train_test_imputation(X_train, X_test)
        
        # Apply skewness correction
        with st.spinner("Correcting skewness..."):
            X_train_corrected, X_test_corrected, transformers = apply_skewness_correction(
                X_train_imp, X_test_imp, binary_cols, time_cols
            )
        
        # Logistic Regression
        st.header("1. Logistic Regression Model")
        
        # Baseline model
        st.subheader("Baseline Model (All Features)")
        col1, col2 = st.columns(2)
        
        with col1:
            lr_baseline = LogisticRegression(max_iter=2000, random_state=2025, solver='lbfgs')
            lr_baseline.fit(X_train_corrected, Y_train)
            y_pred_baseline = lr_baseline.predict(X_test_corrected)
            
            accuracy_baseline = accuracy_score(Y_test, y_pred_baseline)
            f1_baseline = f1_score(Y_test, y_pred_baseline)
            
            st.metric("Accuracy", f"{accuracy_baseline:.4f}")
            st.metric("F1 Score", f"{f1_baseline:.4f}")
        
        with col2:
            # Confusion matrix
            cm_baseline = confusion_matrix(Y_test, y_pred_baseline)
            
            fig_cm = px.imshow(cm_baseline, 
                              text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              x=['No CVD', 'CVD'],
                              y=['No CVD', 'CVD'],
                              title='Confusion Matrix - Baseline',
                              color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature Selection with RFE
        st.subheader("Feature Selection with Recursive Feature Elimination (RFE)")


        @st.cache_resource(show_spinner=True, show_time=True)
        def recursive_feature_elimination(X):
            X_train_corrected = X
            # Generate seeds for robust feature selection
            rng = np.random.RandomState(seed=2025)
            seeds = rng.randint(low=0, high=10000, size=7)

            votes = np.zeros(X_train_corrected.shape[1], dtype=int)

            for seed in seeds:
                model = LogisticRegression(random_state=seed, max_iter=2000, solver='lbfgs')
                rfe = RFE(estimator=model, n_features_to_select=10)
                rfe.fit(X_train_corrected, Y_train)
                votes += rfe.support_.astype(int)

            # Select features that got at least one vote
            mask = votes > 0
            top_features = X_train_corrected.columns[mask].tolist()
            return top_features
        model = LogisticRegression()
        top_features = recursive_feature_elimination(X_train_corrected)
        st.info(f"Selected {len(top_features)} features: {', '.join(top_features[:10])}...")
        
        # Train model with selected features
        st.subheader("Model with Selected Features")
        
        X_train_selected = X_train_corrected[top_features]
        X_test_selected = X_test_corrected[top_features]
        
        col1, col2 = st.columns(2)
        
        with col1:
            lr_selected = LogisticRegression(max_iter=2000, random_state=2025, solver='lbfgs')
            lr_selected.fit(X_train_selected, Y_train)
            y_pred_selected = lr_selected.predict(X_test_selected)
            
            accuracy_selected = accuracy_score(Y_test, y_pred_selected)
            f1_selected = f1_score(Y_test, y_pred_selected)
            
            st.metric("Accuracy", f"{accuracy_selected:.4f}", 
                     delta=f"{accuracy_selected - accuracy_baseline:.4f}")
            st.metric("F1 Score", f"{f1_selected:.4f}",
                     delta=f"{f1_selected - f1_baseline:.4f}")
            
            # Feature importance visualization
            feature_importance = pd.DataFrame({
                'Feature': top_features,
                'Coefficient': lr_selected.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            st.subheader("Top Feature Importance")
            
            # Color code by risk factor (positive = risk, negative = protective)
            colors = ['#d62728' if x > 0 else '#1f77b4' for x in feature_importance['Coefficient']]
            
            fig_importance = go.Figure(data=[
                go.Bar(x=feature_importance['Coefficient'], 
                      y=feature_importance['Feature'],
                      orientation='h',
                      marker=dict(color=colors))
            ])
            fig_importance.update_layout(
                title="Feature Importance (Red: Risk Factor, Blue: Protective)",
                xaxis_title="Coefficient Impact",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Confusion matrix for selected features
            cm_selected = confusion_matrix(Y_test, y_pred_selected)
            
            fig_cm_selected = px.imshow(cm_selected, 
                                       text_auto=True,
                                       labels=dict(x="Predicted", y="Actual"),
                                       x=['No CVD', 'CVD'],
                                       y=['No CVD', 'CVD'],
                                       title='Confusion Matrix - Selected Features',
                                       color_continuous_scale='Blues')
            st.plotly_chart(fig_cm_selected, use_container_width=True)
        
        # Correlation-based feature reduction
        st.subheader("Correlation-Based Feature Reduction")
        
        correlation_threshold = st.slider("Select correlation threshold", 0.8, 0.99, 0.90, 0.05)
        
        corr_matrix = X_train_corrected.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        
        if len(to_drop_corr) > 0:
            st.info(f"Dropping {len(to_drop_corr)} highly correlated features: {', '.join(to_drop_corr)}")
            
            current_features = [f for f in top_features if f not in to_drop_corr]
            X_train_corr_reduced = X_train_corrected[current_features]
            X_test_corr_reduced = X_test_corrected[current_features]
            
            lr_corr = LogisticRegression(max_iter=2000, random_state=2025, solver='lbfgs')
            lr_corr.fit(X_train_corr_reduced, Y_train)
            y_pred_corr = lr_corr.predict(X_test_corr_reduced)
            
            f1_corr = f1_score(Y_test, y_pred_corr)
            accuracy_corr = accuracy_score(Y_test, y_pred_corr)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy (Corr. Reduced)", f"{accuracy_corr:.4f}")
            with col2:
                st.metric("F1 Score (Corr. Reduced)", f"{f1_corr:.4f}")
        else:
            st.success("No features exceed the correlation threshold.")
        
        # Decision Tree
        st.header("2. Decision Tree Classifier")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dt_model = DecisionTreeClassifier(criterion='entropy', random_state=2025, 
                                             class_weight='balanced')
            # Remove TIME variables from training data
            train_cols = [col for col in X_train_corrected.columns if not col.startswith('TIME')]
            dt_model.fit(X_train_corrected[train_cols], Y_train)
            y_pred_dt = dt_model.predict(X_test_corrected[train_cols])
            
            accuracy_dt = accuracy_score(Y_test, y_pred_dt)
            f1_dt = f1_score(Y_test, y_pred_dt)
            
            st.metric("Accuracy", f"{accuracy_dt:.4f}")
            st.metric("F1 Score", f"{f1_dt:.4f}")
            
            # Feature importance
            feature_importance_dt = pd.DataFrame({
                'Feature': X_train_corrected.columns,
                'Importance': dt_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Feature Importance")
            fig_importance_dt = px.bar(feature_importance_dt.head(10), 
                                      x='Importance', y='Feature',
                                      orientation='h',
                                      title='Top 10 Important Features')
            st.plotly_chart(fig_importance_dt, use_container_width=True)
        
        with col2:
            # Confusion matrix
            cm_dt = confusion_matrix(Y_test, y_pred_dt)
            
            fig_cm_dt = px.imshow(cm_dt, 
                                 text_auto=True,
                                 labels=dict(x="Predicted", y="Actual"),
                                 x=['No CVD', 'CVD'],
                                 y=['No CVD', 'CVD'],
                                 title='Confusion Matrix - Decision Tree',
                                 color_continuous_scale='Greens')
            st.plotly_chart(fig_cm_dt, use_container_width=True)
        
        # Model Comparison
        st.header("3. Model Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': ['Baseline LR', 'Selected Features LR', 'Decision Tree'],
            'Accuracy': [accuracy_baseline, accuracy_selected, accuracy_dt],
            'F1 Score': [f1_baseline, f1_selected, f1_dt]
        })
        
        fig_comparison = px.bar(comparison_df, x='Model', y=['Accuracy', 'F1 Score'],
                               barmode='group',
                               title='Model Performance Comparison',
                               labels={'value': 'Score', 'variable': 'Metric'})
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Display best model
        best_model_idx = comparison_df['F1 Score'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]['Model']
        best_f1 = comparison_df.iloc[best_model_idx]['F1 Score']
        
        st.success(f"Best Model: **{best_model}** with F1 Score of **{best_f1:.4f}**")
    else:
        st.warning("CVD column not available in the dataset for machine learning analysis.")

# Conclusion Section
elif selected == 'Conclusion':
    st.title("Conclusion")
    
    st.markdown("""
    ## Conclusions
    """)
    
    st.info("Navigate through the different sections using the sidebar to explore detailed analysis and visualizations.")
