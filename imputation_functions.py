import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def drop_high_missing_cols(df_train, df_test, threshold=0.50):
    """
    1. Drops columns based on missing value percentage in the TRAINING set.
       Applies the same drop to the Test set.
    """
    print(f"--- Running Step 1: Dropping columns > {threshold:.0%} missing ---")

    # Calculate missing percentage on TRAIN only
    missing_pct = df_train.isnull().mean()

    # Identify columns to drop based on training data
    cols_to_drop = missing_pct[missing_pct >= threshold].index.tolist()

    if cols_to_drop:
        print(f"   -> Dropping {len(cols_to_drop)} columns based on Train data: {', '.join(cols_to_drop)}")
        # Drop columns from both Train and Test
        df_train_dropped = df_train.drop(columns=cols_to_drop)

        # Only drop columns from test that actually exist in test (safety check)
        test_cols_to_drop = [c for c in cols_to_drop if c in df_test.columns]
        df_test_dropped = df_test.drop(columns=test_cols_to_drop)
    else:
        print("   -> No columns exceeded the missing value threshold in Train.")
        df_train_dropped = df_train.copy()
        df_test_dropped = df_test.copy()

    return df_train_dropped, df_test_dropped


# --- Helper Function for KNN ---

def _reconstruct_dataframe(encoded_df, original_df, num_cols, cat_cols, enc):
    """
    Helper to reverse One-Hot Encoding.
    """
    # 1. Isolate Numerical Data
    present_num_cols = [col for col in num_cols if col in encoded_df.columns]
    imputed_numerical = encoded_df[present_num_cols]

    # 2. Isolate Categorical Data to Inverse Transform
    if cat_cols:
        encoded_cols_names = enc.get_feature_names_out(cat_cols)

        # Ensure all expected encoded columns exist (fill 0 if missing, e.g., in Test)
        missing_enc_cols = [c for c in encoded_cols_names if c not in encoded_df.columns]
        if missing_enc_cols:
            for c in missing_enc_cols:
                encoded_df[c] = 0

        imputed_encoded = encoded_df[encoded_cols_names]

        # Inverse transform
        imputed_categorical_array = enc.inverse_transform(imputed_encoded)
        imputed_categorical = pd.DataFrame(imputed_categorical_array,
                                           columns=cat_cols,
                                           index=encoded_df.index)
    else:
        imputed_categorical = pd.DataFrame(index=encoded_df.index)

    # 3. Concatenate
    reconstructed_df = pd.concat([imputed_numerical, imputed_categorical], axis=1)

    # 4. Enforce Original Column Order
    original_cols_present = original_df.columns.intersection(reconstructed_df.columns)
    final_df = reconstructed_df.reindex(columns=original_cols_present)

    return final_df


def knn_impute(df_train, df_test, min_thresh=0.05, max_thresh=0.50, n_neighbors=3):
    """
    2. Uses KNN Imputation.
       - Fits Encoder on TRAIN.
       - Trains KNN on TRAIN.
       - Imputes missing values in TRAIN and TEST using the TRAIN-fitted models.
    """
    print(f"\n--- Running Step 2: KNN Imputation ({min_thresh:.0%} - {max_thresh:.0%} missing) ---")

    train_imputed = df_train.copy()
    test_imputed = df_test.copy()

    # --- 1. One-Hot Encoding (Fit on Train, Transform Both) ---

    categorical_cols = [col for col in train_imputed.columns if train_imputed[col].dtype == 'object']
    numerical_cols = [col for col in train_imputed.columns if train_imputed[col].dtype != 'object']

    encoder = None

    if categorical_cols:
        print(f"   -> Fitting OneHotEncoder on {len(categorical_cols)} columns (Train data)...")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Fit on Train
        encoded_train_array = encoder.fit_transform(train_imputed[categorical_cols])
        encoded_train_df = pd.DataFrame(encoded_train_array,
                                        columns=encoder.get_feature_names_out(categorical_cols),
                                        index=train_imputed.index)

        # Transform Test
        encoded_test_array = encoder.transform(test_imputed[categorical_cols])
        encoded_test_df = pd.DataFrame(encoded_test_array,
                                       columns=encoder.get_feature_names_out(categorical_cols),
                                       index=test_imputed.index)

        # Join encoded columns, drop original cats
        df_train_enc = train_imputed.drop(columns=categorical_cols).join(encoded_train_df)
        df_test_enc = test_imputed.drop(columns=categorical_cols).join(encoded_test_df)
    else:
        df_train_enc = train_imputed.copy()
        df_test_enc = test_imputed.copy()

    # --- 2. KNN Imputation ---

    # Calculate means on TRAIN only to fill features for stability
    feature_fill_values = df_train_enc.mean()

    # Iterate over columns in TRAIN to decide what to impute
    for col in df_train_enc.columns:
        missing_pct = df_train_enc[col].isnull().mean()

        # Check eligibility based on Train statistics
        if min_thresh <= missing_pct <= max_thresh:
            print(f"   -> KNN Imputing '{col}' (Train Missing: {missing_pct:.2%})")

            other_cols = [c for c in df_train_enc.columns if c != col]

            # -- Training Data Setup --
            train_rows_not_null = df_train_enc[col].notnull()
            train_rows_null = df_train_enc[col].isnull()

            if not train_rows_null.any() and df_test_enc[col].isnull().sum() == 0:
                print(f"      -> Skipping '{col}', no missing rows in Train or Test.")
                continue

            # X_train_model: The data we teach KNN with (from Train set)
            X_train_model = df_train_enc.loc[train_rows_not_null, other_cols].fillna(feature_fill_values)
            y_train_model = df_train_enc.loc[train_rows_not_null, col]

            # -- Model Selection & Training --
            is_binary = y_train_model.nunique() <= 2
            model = KNeighborsClassifier(n_neighbors=n_neighbors) if is_binary else KNeighborsRegressor(
                n_neighbors=n_neighbors)

            model.fit(X_train_model, y_train_model)

            # -- Prediction Phase --

            # 1. Predict for TRAIN set missing values
            if train_rows_null.any():
                X_predict_train = df_train_enc.loc[train_rows_null, other_cols].fillna(feature_fill_values)
                df_train_enc.loc[train_rows_null, col] = model.predict(X_predict_train)
                print(f"      -> Imputed {train_rows_null.sum()} rows in Train.")

            # 2. Predict for TEST set missing values (Using model trained on Train)
            test_rows_null = df_test_enc[col].isnull()
            if test_rows_null.any():
                X_predict_test = df_test_enc.loc[test_rows_null, other_cols].fillna(feature_fill_values)
                df_test_enc.loc[test_rows_null, col] = model.predict(X_predict_test)
                print(f"      -> Imputed {test_rows_null.sum()} rows in Test.")

    # --- 3. Reconstruct DataFrame ---
    if encoder is None:
        return df_train_enc, df_test_enc

    print("   -> Reconstructing dataframes...")
    final_train = _reconstruct_dataframe(df_train_enc, df_train, numerical_cols, categorical_cols, encoder)
    final_test = _reconstruct_dataframe(df_test_enc, df_test, numerical_cols, categorical_cols, encoder)

    return final_train, final_test


def impute_simple_central(df_train, df_test, max_thresh=0.05):
    """
    3. Simple Imputation (Median/Mode).
       Learns values from TRAIN, applies to TRAIN and TEST.
    """
    print(f"\n--- Running Step 3: Simple Imputation (< {max_thresh:.0%} missing) ---")

    train_imputed = df_train.copy()
    test_imputed = df_test.copy()

    for col in train_imputed.columns:
        # Check missing percentage on Train
        missing_pct = train_imputed[col].isnull().mean()

        if 0 < missing_pct < max_thresh:
            print(f"   -> Processing '{col}' (Train Missing: {missing_pct:.2%})")

            # Determine fill value from Train
            if pd.api.types.is_numeric_dtype(train_imputed[col]):
                fill_value = train_imputed[col].median()
                val_type = "median"
            else:
                fill_value = train_imputed[col].mode()[0]
                val_type = "mode"

            # Fill Train
            train_imputed[col].fillna(fill_value, inplace=True)

            # Fill Test (with Train's value)
            if col in test_imputed.columns:
                test_imputed[col].fillna(fill_value, inplace=True)

            print(f"      -> Filled Train and Test with {val_type}: {fill_value}")

    return train_imputed, test_imputed


# --- Main Orchestrator ---

def process_features(X_train, X_test):
    """
    Orchestrates the feature cleaning pipeline.
    """
    print("==================================================")
    print("STARTING FEATURE IMPUTATION")
    print("==================================================")

    # 1. Drop High Missing
    X_train, X_test = drop_high_missing_cols(X_train, X_test, threshold=0.50)

    # 2. KNN Impute
    X_train, X_test = knn_impute(X_train, X_test, min_thresh=0.05, max_thresh=0.50)

    # 3. Simple Impute
    X_train, X_test = impute_simple_central(X_train, X_test, max_thresh=0.05)

    print("\n--- Feature Imputation Complete ---")
    return X_train, X_test