def distplots(df):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    numeric_df = df

    # Technically not needed but might as well
    numeric_df_nona = numeric_df.dropna()
    column = numeric_df.name

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(numeric_df_nona, bins=30, alpha=0.7, edgecolor='black')

    if len(numeric_df_nona) > 1:
        density = stats.gaussian_kde(numeric_df_nona)
        xs = np.linspace(numeric_df_nona.min(), numeric_df_nona.max(), 200)
        ax.plot(xs, density(xs) * len(numeric_df_nona) * (numeric_df_nona.max() - numeric_df_nona.min()) / 30,
                'r-', linewidth=2)

    ax.set_xlabel(column)
    ax.set_ylabel('Number of Patients')
    ax.set_title(f'Distribution of {column}')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    return(fig)