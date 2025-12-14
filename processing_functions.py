import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def drop_high_correlation(df, threshold=0.90):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping {len(to_drop)} columns due to correlation > {threshold}: {to_drop}")
    return df.drop(columns=to_drop)

def apply_skewness_correction(X_train, X_test, binary_cols, time_cols):
    """
    Apply PowerTransformer (Yeo-Johnson) to correct skewness in numeric columns.
    Excludes binary and time columns.
    """
    X_train_corrected = X_train.copy()
    X_test_corrected = X_test.copy()
    
    # Identify columns to exclude
    cols_to_exclude = list(binary_cols) + list(time_cols)
    numeric_cols = [c for c in X_train.columns if c not in cols_to_exclude]
    
    transformers = {}
    
    for column in numeric_cols:
        skewness = stats.skew(X_train_corrected[column])
        
        if abs(skewness) >= 0.5:
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            col_train = X_train_corrected[[column]]
            col_test = X_test_corrected[[column]]
            
            pt.fit(col_train)
            X_train_corrected[column] = pt.transform(col_train).flatten()
            X_test_corrected[column] = pt.transform(col_test).flatten()
            
            transformers[column] = pt
    
    return X_train_corrected, X_test_corrected, transformers

def train_test_imputation(X_train, X_test, threshold=0.50, range_thresh=(0.02, 0.50)):
    """
    Perform imputation on train and test sets separately, learning from training data.
    
    Steps:
    1. Drop columns with >50% missing in train
    2. Simple imputation for <2% missing
    3. KNN imputation for 2-50% missing
    """
    # 1. Drop columns with more than threshold% missing data (based on Train)
    train_missing_pct = X_train.isnull().mean()
    cols_to_drop = train_missing_pct[train_missing_pct > threshold].index.tolist()
    
    X_train_imp = X_train.drop(columns=cols_to_drop)
    X_test_imp = X_test.drop(columns=cols_to_drop)
    
    # 2. Simple Imputation (learn from train, apply to both)
    simple_impute_cols = train_missing_pct[(train_missing_pct > 0) & (train_missing_pct < range_thresh[0])].index.tolist()
    simple_impute_cols = [c for c in simple_impute_cols if c not in cols_to_drop]
    
    knn_impute_cols = train_missing_pct[
        (train_missing_pct >= range_thresh[0]) & (train_missing_pct <= range_thresh[1])].index.tolist()
    knn_impute_cols = [c for c in knn_impute_cols if c not in cols_to_drop]
    
    # Apply Simple Imputation
    for col in simple_impute_cols:
        if pd.api.types.is_numeric_dtype(X_train_imp[col]):
            fill_val = X_train_imp[col].median()
        else:
            mode_result = X_train_imp[col].mode()
            if len(mode_result) > 0:
                fill_val = mode_result[0]
            else:
                # If mode is empty, try value_counts
                value_counts = X_train_imp[col].value_counts()
                if len(value_counts) > 0:
                    fill_val = value_counts.index[0]
                else:
                    # If all values are NaN, skip this column
                    print(f"Warning: Column '{col}' has all NaN values, skipping imputation")
                    continue
        
        X_train_imp[col] = X_train_imp[col].fillna(fill_val)
        X_test_imp[col] = X_test_imp[col].fillna(fill_val)
    
    # 3. Apply KNN Imputation
    categorical_cols = [col for col in X_train_imp.columns if X_train_imp[col].dtype == 'object']
    numerical_cols = [col for col in X_train_imp.columns if X_train_imp[col].dtype != 'object']
    
    # Encoder for categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if categorical_cols:
        # Fit on Train
        encoder.fit(X_train_imp[categorical_cols])
        
        # Transform Train
        encoded_train = pd.DataFrame(encoder.transform(X_train_imp[categorical_cols]),
                                     columns=encoder.get_feature_names_out(categorical_cols),
                                     index=X_train_imp.index)
        X_train_encoded = X_train_imp.drop(columns=categorical_cols).join(encoded_train)
        
        # Transform Test
        encoded_test = pd.DataFrame(encoder.transform(X_test_imp[categorical_cols]),
                                    columns=encoder.get_feature_names_out(categorical_cols),
                                    index=X_test_imp.index)
        X_test_encoded = X_test_imp.drop(columns=categorical_cols).join(encoded_test)
    else:
        X_train_encoded = X_train_imp.copy()
        X_test_encoded = X_test_imp.copy()
    
    # Fill temporary NaNs with Train means
    train_means = X_train_encoded.mean()
    X_train_filled_features = X_train_encoded.fillna(train_means)
    X_test_filled_features = X_test_encoded.fillna(train_means)
    
    n_neighbors = 5
    
    for col in knn_impute_cols:
        is_categorical = col in categorical_cols
        
        if col in categorical_cols:
            col_encoded_names = [c for c in X_train_encoded.columns if c.startswith(f"{col}_") or c == col]
            cols_to_exclude = col_encoded_names
        else:
            cols_to_exclude = [col]
        
        feature_cols = [c for c in X_train_encoded.columns if c not in cols_to_exclude]
        
        train_indices = X_train_imp[col].notnull()
        X_train_features = X_train_filled_features.loc[train_indices, feature_cols]
        y_train_target = X_train_imp.loc[train_indices, col]
        
        predict_train_indices = X_train_imp[col].isnull()
        X_predict_train = X_train_filled_features.loc[predict_train_indices, feature_cols]
        
        predict_test_indices = X_test_imp[col].isnull()
        X_predict_test = X_test_filled_features.loc[predict_test_indices, feature_cols]
        
        if len(X_predict_train) == 0 and len(X_predict_test) == 0:
            continue
        
        if is_categorical or X_train_imp[col].nunique() <= 2:
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        model.fit(X_train_features, y_train_target)
        
        if len(X_predict_train) > 0:
            X_train_imp.loc[predict_train_indices, col] = model.predict(X_predict_train)
        
        if len(X_predict_test) > 0:
            X_test_imp.loc[predict_test_indices, col] = model.predict(X_predict_test)
    
    return X_train_imp, X_test_imp