import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import PowerTransformer
from streamlit_option_menu import option_menu
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from processing_functions import train_test_imputation, apply_skewness_correction
from sklearn.linear_model import LogisticRegression
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
        options=['Abstract', 'Data Preprocessing', 'Exploratory Data Analysis', 'Statistical Analysis', 'Machine Learning Results', 'Conclusion'],
        menu_icon='heart-pulse',
        icons=['bookmark-check', 'wrench', 'bar-chart', 'calculator', 'cpu', 'check2-circle'],
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

# Data Preprocessing Section
elif selected == 'Data Preprocessing':
    st.title("Data Preprocessing")
    
    st.markdown("""
    This section describes the data preprocessing pipeline used to prepare 
    the Framingham Heart Study dataset for analysis and machine learning.
    """)

    # Train-Test Split Strategy
    st.header("0. Train-Test Split Strategy")

    st.markdown("""
        To prevent data leakage, the preprocessing pipeline follows a strict order:
        """)

    st.code("""
    1. Split data (80% train, 20% test) with stratification
    2. Learn imputation parameters from TRAINING set only
    3. Apply learned parameters to both train and test sets
    4. Transform features (skewness correction) on train, then test
    5. Select features based on training set performance""", language="text")

    # Missing Data Analysis
    st.header("1. Missing Data Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Calculate missing percentages
        missing_pct = (data_raw.isnull().sum() / len(data_raw) * 100).sort_values(ascending=False)
        missing_df = pd.DataFrame({
            'Column': missing_pct.index,
            'Missing Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Percentage'] > 0]
        
        if not missing_df.empty:
            fig_missing = px.bar(missing_df, 
                               x='Missing Percentage', 
                               y='Column',
                               orientation='h',
                               title='Missing Data by Feature',
                               labels={'Missing Percentage': 'Missing (%)'})
            fig_missing.update_layout(height=400)
            st.plotly_chart(fig_missing, width="stretch")
        else:
            st.info("No missing data detected in the dataset.")
    
    with col2:
        st.subheader("Missing Data Handling")
        st.markdown("""
        **Three-tier approach:**
        
        1. **Drop columns** with >50% missing data
        2. **KNN Imputation** for 2-50% missing data
        3. **Simple Imputation** (median/mode) for <2% missing data
        """)
        
        if not missing_df.empty:
            high_missing = missing_df[missing_df['Missing Percentage'] > 50]
            moderate_missing = missing_df[(missing_df['Missing Percentage'] >= 2) & 
                                         (missing_df['Missing Percentage'] <= 50)]
            low_missing = missing_df[missing_df['Missing Percentage'] < 2]
            
            st.metric("Columns Dropped (>50%)", len(high_missing))
            st.metric("KNN Imputed (2-50%)", len(moderate_missing))
            st.metric("Simple Imputed (<2%)", len(low_missing))
    
    # Column Dropping
    st.header("2. Dropping High-Missing Columns")
    
    st.markdown("""
    Columns with more than 50% missing values were removed from the dataset as they 
    provide insufficient information for reliable analysis or imputation.
    """)
    
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    if high_missing_cols:
        st.warning(f"**Columns dropped:** {', '.join(high_missing_cols)}")
        st.code(f"Threshold: 50% missing\nColumns affected: {len(high_missing_cols)}")
    else:
        st.success("No columns exceeded the 50% missing threshold.")
    
    # KNN Imputation
    st.header("3. KNN Imputation")
    
    st.markdown("""
    For columns with moderate missingness (2-50%), K-Nearest Neighbors (KNN) imputation 
    was used. The KNN imputer was trained on the X-train split, and then applied to the X-test and X-train split
    in order to prevent data leakage from an improperly trained imputer.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("How Our KNN Imputation Works")
        st.markdown("""
        1. **One-hot encode** categorical variables
        2. For each column with missing data:
           - Use complete cases as training data
           - Find k=5 nearest neighbors
           - Predict missing values using:
             - **KNN Regressor** for continuous variables
             - **KNN Classifier** for binary variables
        3. **Reconstruct** original data format
        """)
    
    with col2:
        knn_impute_cols = missing_pct[(missing_pct >= 2) & (missing_pct <= 50)].index.tolist()
        knn_impute_cols = [c for c in knn_impute_cols if c not in high_missing_cols]
        
        if knn_impute_cols:
            st.subheader("Columns Using KNN")
            for col in knn_impute_cols:
                st.text(f"{col}: {missing_pct[col]:.2f}% missing")
        else:
            st.info("No columns in the 2-50% missing range.")
    
    # Simple Imputation
    st.header("4. Simple Imputation")
    
    st.markdown("""
    For columns with minimal missingness (<2%), simple central tendency imputation is used:
    - **Median** for numerical variables
    - **Mode** for categorical variables
    """)
    
    simple_impute_cols = missing_pct[(missing_pct > 0) & (missing_pct < 2)].index.tolist()
    simple_impute_cols = [c for c in simple_impute_cols if c not in high_missing_cols]
    
    if simple_impute_cols:
        st.info(f"**Columns with simple imputation:** {', '.join(simple_impute_cols)}")
    else:
        st.success("No columns require simple imputation.")
    
    # Distribution Comparison: Before and After Imputation
    st.header("5. Distribution Comparison: Before vs After Imputation")
    
    st.markdown("""
    Comparing distributions before and after imputation helps verify that the imputation 
    process maintains the statistical properties of the data.
    """)
    
    # Select columns that had imputation applied
    cols_with_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 50)].index.tolist()
    cols_with_missing = [c for c in cols_with_missing if c in data_raw.columns and c in data_imputed.columns]
    
    if cols_with_missing:
        # Let user select a column to compare
        selected_comparison_col = st.selectbox(
            "Select a column to compare distributions:",
            cols_with_missing,
            index=0 if cols_with_missing else None
        )
        
        if selected_comparison_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Before Imputation")
                fig_before = go.Figure()
                
                # Histogram
                fig_before.add_trace(go.Histogram(
                    x=data_raw[selected_comparison_col].dropna(),
                    nbinsx=30,
                    name='Histogram',
                    marker_color='lightblue'
                ))
                
                # KDE
                values_before = data_raw[selected_comparison_col].dropna()
                if len(values_before) > 1:
                    kde_x = np.linspace(values_before.min(), values_before.max(), 100)
                    kde = stats.gaussian_kde(values_before)
                    kde_y = kde(kde_x)
                    fig_before.add_trace(go.Scatter(
                        x=kde_x,
                        y=kde_y * len(values_before) * (values_before.max() - values_before.min()) / 30,
                        mode='lines',
                        name='KDE',
                        line=dict(color='red', width=2)
                    ))
                
                fig_before.update_layout(
                    title=f'{selected_comparison_col} - Before Imputation',
                    xaxis_title=selected_comparison_col,
                    yaxis_title='Count',
                    showlegend=False,
                    height=350
                )
                st.plotly_chart(fig_before, width="stretch")
                
                # Statistics before
                st.metric("Count (non-null)", f"{data_raw[selected_comparison_col].notna().sum():,}")
                st.metric("Missing", f"{data_raw[selected_comparison_col].isna().sum():,} ({missing_pct[selected_comparison_col]:.2f}%)")
            
            with col2:
                st.subheader("After Imputation")
                fig_after = go.Figure()
                
                # Histogram
                fig_after.add_trace(go.Histogram(
                    x=data_imputed[selected_comparison_col],
                    nbinsx=30,
                    name='Histogram',
                    marker_color='lightgreen'
                ))
                
                # KDE
                values_after = data_imputed[selected_comparison_col].dropna()
                if len(values_after) > 1:
                    kde_x = np.linspace(values_after.min(), values_after.max(), 100)
                    kde = stats.gaussian_kde(values_after)
                    kde_y = kde(kde_x)
                    fig_after.add_trace(go.Scatter(
                        x=kde_x,
                        y=kde_y * len(values_after) * (values_after.max() - values_after.min()) / 30,
                        mode='lines',
                        name='KDE',
                        line=dict(color='darkgreen', width=2)
                    ))
                
                fig_after.update_layout(
                    title=f'{selected_comparison_col} - After Imputation',
                    xaxis_title=selected_comparison_col,
                    yaxis_title='Count',
                    showlegend=False,
                    height=350
                )
                st.plotly_chart(fig_after, width="stretch")
                
                # Statistics after
                st.metric("Count (non-null)", f"{data_imputed[selected_comparison_col].notna().sum():,}")
                st.metric("Missing", f"{data_imputed[selected_comparison_col].isna().sum():,}")
    else:
        st.info("No columns with missing data to compare.")
    
    # Correlation Comparison: Before and After Imputation
    st.header("5b. Correlation Comparison: Before vs After Imputation")
    
    st.markdown("""
    Comparing correlation matrices before and after imputation reveals how imputation 
    affects relationships between variables. Proper imputation should preserve or enhance 
    existing correlations without introducing spurious relationships.
    """)
    
    # Select key numeric variables for correlation comparison
    available_vars_for_corr = data_imputed.columns
    if len(available_vars_for_corr) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before Imputation")
            # Calculate correlation only on complete cases
            corr_before = data_raw[available_vars_for_corr].corr()
            
            fig_corr_before = px.imshow(
                corr_before,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title='Correlation Matrix - Before Imputation'
            )
            fig_corr_before.update_layout(height=500)
            st.plotly_chart(fig_corr_before, width="stretch")
        
        with col2:
            st.subheader("After Imputation")
            # Calculate correlation on imputed data
            corr_after = data_imputed[available_vars_for_corr].corr()
            
            fig_corr_after = px.imshow(
                corr_after,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title='Correlation Matrix - After Imputation'
            )
            fig_corr_after.update_layout(height=500)
            st.plotly_chart(fig_corr_after, width="stretch")
        
        # Calculate and display correlation differences
        st.subheader("Correlation Changes")
        df_a = data_raw.drop(columns=['HDLC', 'LDLC'])
        df_b = data_imputed

        corr_a = df_a.corr()
        corr_b = df_b.corr()

        common = corr_a.columns.intersection(corr_b.columns)
        diff = corr_b.loc[common, common] - corr_a.loc[common, common]

        # Calculate and display correlation differences
        fig_corr_diff = px.imshow(
            diff,
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0,
            aspect="square",
            title="Correlation Difference: Original vs imputed (without dropped variables)"
        )

        # Update the layout to match the original styling
        fig_corr_diff.update_layout(
            xaxis=dict(tickangle=90),  # Rotate x-axis labels 90 degrees
            yaxis=dict(tickangle=0),  # Keep y-axis labels horizontal
            coloraxis_colorbar=dict(
                title="Correlation difference"  # Add colorbar label
            ),
        )
        fig_corr_after.update_layout(width=600)
        st.plotly_chart(fig_corr_diff, width="stretch", height=600)

        st.markdown("""
        **Legend:**
        - **[Red] Positive values:** Correlation increased after imputation
        - **[Blue] Negative values:** Correlation decreased after imputation
        - **[White] Zero or close to zero:** Little to no change in correlation
        """)
    else:
        st.warning("Insufficient variables available for correlation comparison.")

    # Outlier Detection
    st.header("6. Outlier Detection")

    st.markdown("""
    Outliers were detected using the Interquartile Range (IQR) method. While outliers 
    were identified, they were kept, as they may represent genuine extreme 
    cases in cardiovascular health data. Such cases may represent diseased cases that we would like our model to also be able to identify.
    """)

    numeric_cols_raw = data_raw.select_dtypes(include=['number']).columns.tolist()

    # Add selectbox for variable selection
    example_col = st.selectbox("Select variable for outlier detection:", numeric_cols_raw)

    # Calculate outliers for selected column
    Q1 = data_raw[example_col].quantile(0.25)
    Q3 = data_raw[example_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data_raw[(data_raw[example_col] < lower_bound) |
                        (data_raw[example_col] > upper_bound)][example_col]

    col1, col2 = st.columns([3, 2])

    with col1:
        # Box plot showing outliers
        fig_outlier = px.box(data_raw, y=example_col,
                             title=f'Box-and-whiskers plot of {example_col}')
        st.plotly_chart(fig_outlier, width="stretch")

    with col2:
        st.subheader("IQR Method")
        st.code(f"""Q1 = {Q1:.2f}
    Q3 = {Q3:.2f}
    IQR = {IQR:.2f}
    Lower Bound = {lower_bound:.2f}
    Upper Bound = {upper_bound:.2f}
    Outliers: {len(outliers)} ({len(outliers) / len(data_raw) * 100:.2f}%)""")

    # Skewness Analysis
    st.header("7. Skewness Analysis and Correction")
    
    st.markdown("""
    Skewness affects model performance. Features with |skewness| > 0.5 are transformed 
    using PowerTransformer (Yeo-Johnson method) to achieve more symmetric distributions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Preprocessing")
        
        # Calculate skewness on raw data
        numeric_data = data_raw.select_dtypes(include=['number'])
        skewness_before = numeric_data.apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
        
        skew_df_before = pd.DataFrame({
            'Feature': skewness_before.index,
            'Skewness': skewness_before.values
        }).head(10)
        
        fig_skew_before = px.bar(skew_df_before, 
                                x='Skewness', 
                                y='Feature',
                                orientation='h',
                                title='Top 10 Skewed Features (Raw Data)',
                                color='Skewness',
                                color_continuous_scale='Reds',
                                color_continuous_midpoint=0,
                                range_color=[0, 10],
                                range_x=[0,10],)
        st.plotly_chart(fig_skew_before, width="stretch")

    with col2:
        st.subheader("After Preprocessing")
        
        # Calculate skewness on imputed data
        numeric_imputed = data_imputed.drop([col for col in data_imputed if col.startswith('TIME')], axis=1).select_dtypes(include=['number'])
        imputed_vars = []
        for column in numeric_imputed:
            skewness = stats.skew(numeric_imputed[column])
            if abs(skewness) >= 0.5:
                pt = PowerTransformer(method='yeo-johnson', standardize=True)
                col_train = numeric_imputed[[column]]
                pt.fit(col_train)
                numeric_imputed[column] = pt.transform(col_train).flatten()
                imputed_vars.append(column)

        skewness_after = numeric_imputed.apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)

        skew_df_after = pd.DataFrame({
            'Feature': skewness_after.index,
            'Skewness': skewness_after.values
        }).head(10)

        fig_skew_after = px.bar(skew_df_after,
                               x='Skewness', 
                               y='Feature',
                               orientation='h',
                               title='Top 10 Skewed Features (After Imputation)',
                               color='Skewness',
                               color_continuous_scale='Reds',
                               color_continuous_midpoint = 0,
                               range_color=[0,10],
                               range_x=[0,10],)
        st.plotly_chart(fig_skew_after, width="stretch")
    st.warning(f"The columns {', '.join(imputed_vars)} were transformed using PowerTransformer (Yeo-Johnson) to reduce skewness.")
    st.markdown("""
    **PowerTransformer (Yeo-Johnson):**
    - Applied to features with |skewness| >= 0.5
    - Excludes binary and time-based columns
    - Standardizes the transformed data
    - Reduces the impact of extreme values
    """)
    
    # Feature Selection
    st.header("8. Feature Selection Methods")
    
    st.markdown("""
    Two complementary feature selection methods are employed in the machine learning pipeline:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation-Based Filtering")
        st.markdown("""
        **Purpose:** Remove redundant features
        
        **Method:**
        - Calculate pairwise correlations
        - Drop features with correlation >0.90
        - Retains one from each highly correlated pair
        
        **Benefit:** Reduces multicollinearity
        """)
    
    with col2:
        st.subheader("Recursive Feature Elimination")
        st.markdown("""
        **Purpose:** Select most predictive features
        
        **Method:**
        - Use Logistic Regression as estimator
        - Run RFE with multiple random seeds (n=7)
        - Select features with at least one vote
        - Target: top 10 features
        
        **Benefit:** Improves model interpretability
        """)
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Set", "80%")
    with col2:
        st.metric("Test Set", "20%")
    with col3:
        st.metric("Stratification", "By CVD outcome")

    # Summary

    # Create a flow diagram
    st.subheader("Preprocessing Flow")

    preprocessing_steps = pd.DataFrame({
        'Step': ['1. Load Data', '2. Train-Test Split', '3. Drop Columns (>50% missing)',
                '4. KNN Imputation (2-50%)', '5. Simple Imputation (<2%)',
                '6. Skewness Correction', '7. Correlation Filter', '8. Feature Selection (RFE)'],
        'Purpose': ['Load raw dataset', 'Stratified 80-20 split', 'Remove low-information features',
                   'Predict moderate missingness', 'Fill minimal missingness',
                   'Normalize distributions', 'Remove redundancy', 'Select best predictors'],
        'Output': [f'{data_raw.shape[0]} rows, {data_raw.shape[1]} cols',
                  'Separate train/test', f'{data_raw.shape[1] - len(high_missing_cols)} cols',
                  'Complete train/test', 'No missing values',
                  'Normalized features', 'Reduced feature set', 'Final feature set']
    })

    st.dataframe(preprocessing_steps, width="stretch", hide_index=True)

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
    st.plotly_chart(fig, width="stretch")

    # Box-whisker plot selector: all variables or selected variable
    box_scope = st.selectbox("Show box-whisker plot for", ["All numeric variables", "Selected variable"], index=0)

    if box_scope == "All numeric variables":
        # Convert to long form and plot one box per numeric variable
        df_long = data_to_use[numeric_cols].melt(var_name='Variable', value_name='Value')
        fig_box_all = px.box(df_long, x='Variable', y='Value', points='outliers',
                             title='Box-Whisker Plots for All Numeric Variables')
        fig_box_all.update_layout(height=600)
        st.plotly_chart(fig_box_all, width="stretch")
    else:
        fig_box = px.box(data_to_use, y=selected_col, title=f'Box-Whisker Plot: {selected_col}')
        st.plotly_chart(fig_box, width="stretch")

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
        st.plotly_chart(fig_corr, width="stretch")

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
            st.plotly_chart(pairplots(key_vars), width="stretch")
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
        st.plotly_chart(fig, width="stretch")
    
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
        st.plotly_chart(fig_sys, width="stretch")
        
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
        st.plotly_chart(fig_dias, width="stretch")
        
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
            st.plotly_chart(fig_age, width="stretch")

# Machine Learning Results Section
elif selected == 'Machine Learning Results':
    st.title("Machine Learning Results")
    
    st.markdown("""
    This section presents the results of machine learning models trained to predict 
    cardiovascular disease outcomes using advanced preprocessing techniques.
    """)
    
    # Prepare data for ML with proper train-test split before imputation

    # Use CVD as the target variable (as in Code.ipynb)
    if 'CVD' in data_raw.columns:
        # First, split the raw data
        CategoryColumn = 'CVD'
        X = data_raw.drop(columns=[CategoryColumn])
        time_cols = [col for col in X.columns if col.startswith('TIME')]
        X = X.drop(columns=time_cols)
        y = data_raw[CategoryColumn]
        
        # Train-test split (80-20) with stratification
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2025)
        
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
            st.plotly_chart(fig_cm, width="stretch")
        
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
            precision_selected = precision_score(Y_test, y_pred_selected)
            recall_selected = recall_score(Y_test, y_pred_selected)

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
            st.plotly_chart(fig_importance, width="stretch")
        
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
            st.plotly_chart(fig_cm_selected, width="stretch")
        st.header("2. Decision Tree Classifier")
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

        else:
            st.success("No features exceed the correlation threshold.")
        
        # Decision Tree

        col1, col2 = st.columns(2)
        
        with col1:
            dt_model = DecisionTreeClassifier(criterion='entropy', random_state=2025, 
                                             class_weight='balanced')
            dt_model.fit(X_train_corrected, Y_train)
            y_pred_dt = dt_model.predict(X_test_corrected)
            
            accuracy_dt = accuracy_score(Y_test, y_pred_dt)
            f1_dt = f1_score(Y_test, y_pred_dt)
            
            st.metric("Accuracy", f"{accuracy_dt:.4f}",
                     delta=f"{accuracy_dt - accuracy_selected:.4f}")

            st.metric("F1 Score", f"{f1_dt:.4f}",
                    delta = f"{f1_dt - f1_selected:.4f}")



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
            st.plotly_chart(fig_importance_dt, width="stretch")
        
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
            st.plotly_chart(fig_cm_dt, width="stretch")
        
        # Random Forest with Hyperparameter Tuning
        st.header("3. Random Forest Classifier with Hyperparameter Tuning")
        
        # Hyperparameter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider(
                "Number of Trees (n_estimators)",
                min_value=10,
                max_value=500,
                value=250,
                step=10,
                help="Number of trees in the forest. More trees generally improve performance but increase computation time."
            )
        
        with col2:
            max_depth = st.slider(
                "Maximum Depth",
                min_value=2,
                max_value=50,
                value=10,
                step=1,
                help="Maximum depth of each tree. Deeper trees can capture more complexity but may overfit."
            )
        
        with col3:
            min_samples_split = st.slider(
                "Min Samples Split",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                help="Minimum number of samples required to split an internal node."
            )
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            min_samples_leaf = st.slider(
                "Min Samples Leaf",
                min_value=1,
                max_value=20,
                value=6,
                step=1,
                help="Minimum number of samples required at a leaf node."
            )
        
        with col5:
            max_features = st.selectbox(
                "Max Features",
                options=['sqrt', 'log2', None],
                index=0,
                help="Number of features to consider when looking for the best split."
            )
        
        with col6:
            class_weight_rf = st.selectbox(
                "Class Weight",
                options=['balanced', 'balanced_subsample', None],
                index=0,
                help="Weights associated with classes to handle imbalanced data."
            )
        
        # Train button
        if st.button("Train Random Forest Model", type="primary"):
            with st.spinner("Training Random Forest model..."):
                # Train Random Forest with selected hyperparameters
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight=class_weight_rf,
                    random_state=2025,
                    n_jobs=-1
                )
                rf_model.fit(X_train_corrected, Y_train)
                y_pred_rf = rf_model.predict(X_test_corrected)
                
                accuracy_rf = accuracy_score(Y_test, y_pred_rf)
                f1_rf = f1_score(Y_test, y_pred_rf)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                st.metric("Accuracy", f"{accuracy_rf:.4f}", delta=f"{accuracy_rf - accuracy_selected:.4f}")
                st.metric("F1 Score", f"{f1_rf:.4f}", delta=f"{f1_rf - f1_selected:.4f}")
                precision_rf = precision_score(Y_test, y_pred_rf)
                recall_rf = recall_score(Y_test, y_pred_rf)
                st.metric("Recall", f"{recall_rf:.4f}", delta=f"{recall_rf - recall_selected:.4f}")
                st.metric("Precision", f"{precision_rf:.4f}", delta=f"{precision_rf - precision_selected:.4f}")

                # Hyperparameters used
                st.subheader("Hyperparameters Used")
                st.code(f"""n_estimators: {n_estimators}
max_depth: {max_depth}
min_samples_split: {min_samples_split}
min_samples_leaf: {min_samples_leaf}
max_features: {max_features}
class_weight: {class_weight_rf}""")
                
                # Feature importance
            with col2:
                # Confusion matrix
                cm_rf = confusion_matrix(Y_test, y_pred_rf)
                
                fig_cm_rf = px.imshow(cm_rf, 
                                     text_auto=True,
                                     labels=dict(x="Predicted", y="Actual"),
                                     x=['No CVD', 'CVD'],
                                     y=['No CVD', 'CVD'],
                                     title='Confusion Matrix - Random Forest',
                                     color_continuous_scale='Oranges')
                st.plotly_chart(fig_cm_rf, width="stretch")

                feature_importance_rf = pd.DataFrame({
                    'Feature': X_train_corrected.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                st.subheader("Top 10 Important Features")
                fig_importance_rf = px.bar(feature_importance_rf.head(10),
                                           x='Importance', y='Feature',
                                           orientation='h',
                                           title='Feature Importance',
                                           color='Importance',
                                           color_continuous_scale='Viridis')
                st.plotly_chart(fig_importance_rf, width="stretch")

            # Store results for comparison
            st.session_state['rf_accuracy'] = accuracy_rf
            st.session_state['rf_f1'] = f1_rf
        
        # Model Comparison
        st.header("4. Model Comparison")
        
        # Build comparison dataframe
        models_list = ['Baseline LR', 'Selected Features LR', 'Decision Tree']
        accuracy_list = [accuracy_baseline, accuracy_selected, accuracy_dt]
        f1_list = [f1_baseline, f1_selected, f1_dt]
        
        # Add Random Forest if it was trained
        if 'rf_accuracy' in st.session_state and 'rf_f1' in st.session_state:
            models_list.append('Random Forest')
            accuracy_list.append(st.session_state['rf_accuracy'])
            f1_list.append(st.session_state['rf_f1'])
        
        comparison_df = pd.DataFrame({
            'Model': models_list,
            'Accuracy': accuracy_list,
            'F1 Score': f1_list
        })
        
        fig_comparison = px.bar(comparison_df, x='Model', y=['Accuracy', 'F1 Score'],
                               barmode='group',
                               title='Model Performance Comparison',
                               labels={'value': 'Score', 'variable': 'Metric'})
        st.plotly_chart(fig_comparison, width="stretch")
        
        # Display best model
        best_model_idx = comparison_df['F1 Score'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]['Model']
        best_f1 = comparison_df.iloc[best_model_idx]['F1 Score']
        
        st.success(f"🏆 Best Model: **{best_model}** with F1 Score of **{best_f1:.4f}**")
    else:
        st.warning("CVD column not available in the dataset for machine learning analysis.")

# Conclusion Section
elif selected == 'Conclusion':
    st.title("Conclusion")
    
    st.markdown("""
    ## Conclusions
    """)
    
    st.info("Navigate through the different sections using the sidebar to explore detailed analysis and visualizations.")
