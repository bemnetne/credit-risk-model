import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_overview(df):
    """
    Display basic dataset information.
    """

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nData Types:")
    print(df.dtypes)

    print("\nDataset Info:")
    print(df.info())

def missing_values_report(df):
    """
    Display missing value counts and percentages.
    """

    missing_count = df.isnull().sum()

    missing_percent = (
        df.isnull().sum() /
        len(df)
    ) * 100

    report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percent
    })

    report = report[
        report["Missing Count"] > 0
    ].sort_values(
        by="Missing Percentage",
        ascending=False
    )

    return report

def check_duplicates(df):
    """
    Count duplicate rows.
    """

    duplicates = df.duplicated().sum()

    print(f"Duplicate Rows: {duplicates}")

    return duplicates

def get_numerical_columns(df):
    """
    Return numerical columns.
    """

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()

def skewness_report(df):
    """
    Calculate skewness.
    """

    numerical_cols = get_numerical_columns(df)

    skewness = (
        df[numerical_cols]
        .skew()
        .sort_values(
            ascending=False
        )
    )

    return skewness

def plot_histograms(
    df,
    bins=30,
    figsize=(15, 10)
):
    """
    Plot histograms for all numerical features.
    """

    numerical_cols = get_numerical_columns(df)

    df[numerical_cols].hist(
        bins=bins,
        figsize=figsize
    )

    plt.tight_layout()
    plt.show()

def categorical_distribution(df):
    """
    Display frequency distribution of categorical features.
    """

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_cols:

        print(f"\n{'='*50}")
        print(f"{col}")
        print(f"{'='*50}")

        print(
            df[col]
            .value_counts()
        )
def plot_categorical_distribution(df):
    """
    Plot distributions of useful categorical features.
    """

    useful_categorical_cols = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "CurrencyCode"
    ]

    # Keep only columns that actually exist in the dataframe
    available_cols = [
        col for col in useful_categorical_cols
        if col in df.columns
    ]

    for col in available_cols:

        plt.figure(figsize=(10, 5))

        order = (
            df[col]
            .value_counts()
            .index
        )

        sns.countplot(
            data=df,
            x=col,
            order=order
        )

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

def correlation_heatmap(
    df,
    figsize=(12, 8)
):
    """
    Plot correlation matrix.
    """

    numerical_cols = get_numerical_columns(df)

    corr = (
        df[numerical_cols]
        .corr()
    )

    plt.figure(
        figsize=figsize
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Matrix"
    )

    plt.show()
def plot_boxplots(df):

    numerical_cols = [
        "Amount",
        "Value",
        # "PricingStrategy",
        # "FraudResult"
    ]
    # numerical_cols = get_numerical_columns(df)
    numerical_cols = [
        col for col in numerical_cols
        if col in df.columns
    ]

    fig, axes = plt.subplots(
        len(numerical_cols),
        1,
        figsize=(12, 4 * len(numerical_cols))
    )

    if len(numerical_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numerical_cols):

        sns.boxplot(
            x=df[col],
            ax=ax
        )

        ax.set_title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

def outlier_report(df):

    numerical_cols = [
        "Amount",
        "Value"
    ]

    report = {}

    for col in numerical_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df[col] < lower) |
            (df[col] > upper)
        ]

        report[col] = {
            "Outlier Count": len(outliers),
            "Outlier Percentage":
                round(
                    len(outliers) /
                    len(df) * 100,
                    2
                )
        }

    return pd.DataFrame(report).T