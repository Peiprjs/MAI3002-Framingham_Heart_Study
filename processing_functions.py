import numpy as np

def drop_high_correlation(df, threshold=0.90):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping {len(to_drop)} columns due to correlation > {threshold}: {to_drop}")
    return df.drop(columns=to_drop)