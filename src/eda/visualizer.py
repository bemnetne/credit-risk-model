import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

class EDAVisualizer:

    def __init__(self, df):
        self.df = df.copy()

    def numerical_distributions(self, columns):

        n_cols = 2
        n_rows = int(np.ceil(len(columns) / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, 4 * n_rows)
        )

        axes = axes.flatten()

        for idx, col in enumerate(columns):

            sns.histplot(
                self.df[col],
                kde=True,
                ax=axes[idx]
            )

            axes[idx].set_title(col)

        plt.tight_layout()
        plt.show()

    def boxplots(self, columns):

        n_cols = 2
        n_rows = int(np.ceil(len(columns) / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, 4 * n_rows)
        )

        axes = axes.flatten()

        for idx, col in enumerate(columns):

            sns.boxplot(
                x=self.df[col],
                ax=axes[idx]
            )

            axes[idx].set_title(col)

        # Remove unused subplots
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def categorical_distribution(self, column):

        plt.figure(figsize=(10, 5))

        sns.countplot(
            data=self.df,
            x=column,
            order=self.df[column]
            .value_counts()
            .index
        )

        plt.xticks(rotation=45)

        plt.title(f"{column} Distribution")

        plt.tight_layout()
        plt.show()
    def categorical_distributions(self):
    
        categorical_cols = [
            "ProductCategory",
            "ChannelId",
            "ProviderId",
            "PricingStrategy"
        ]

        n_cols = 2
        n_rows = math.ceil(len(categorical_cols) / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(16, 5 * n_rows)
        )

        axes = axes.flatten()

        for idx, col in enumerate(categorical_cols):

            sns.countplot(
                data=self.df,
                x=col,
                order=self.df[col].value_counts().index,
                ax=axes[idx]
            )

            axes[idx].set_title(f"{col} Distribution")
            axes[idx].tick_params(axis="x", rotation=45)

        # Remove unused subplots
        for idx in range(len(categorical_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()
    def fraud_distribution(self):

        plt.figure(figsize=(6, 4))

        sns.countplot(
            data=self.df,
            x="FraudResult"
        )

        plt.title("Fraud Distribution")

        plt.show()
    def correlation_heatmap(
        self,
        columns=None
    ):

        if columns is None:
            df_corr = self.df.select_dtypes(
                include="number"
            )
        else:
            df_corr = self.df[columns]

        corr_matrix = df_corr.corr()

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(
            "Correlation Heatmap"
        )

        plt.show()