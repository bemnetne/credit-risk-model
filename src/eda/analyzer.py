import pandas as pd
import numpy as np


class FraudEDA:

    def __init__(self, df):
        self.df = df.copy()

    def dataset_overview(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1]
        }

    def data_types(self):
        return pd.DataFrame(
            self.df.dtypes,
            columns=["dtype"]
        )

    def missing_values(self):
        missing = self.df.isnull().sum()

        return pd.DataFrame({
            "Missing Count": missing,
            "Missing Percentage":
            round((missing / len(self.df)) * 100, 2)
        }).sort_values(
            "Missing Count",
            ascending=False
        )

    def duplicate_count(self):
        return self.df.duplicated().sum()

    def summary_statistics(self):
        return self.df.describe().T

    def skewness_report(self):

        numerical_cols = self.df.select_dtypes(
            include=np.number
        ).columns

        return self.df[
            numerical_cols
        ].skew().sort_values(
            ascending=False
        )
    def get_numerical_columns(df):
        """
        Return numerical columns.
        """

        return df.select_dtypes(
        include=np.number
        ).columns.tolist()
    def correlation_matrix(self, columns=None):
        """
        Compute correlation matrix for selected columns.
        """

        if columns is None:
            df_corr = self.df.select_dtypes(
                include="number"
            )
        else:
            df_corr = self.df[columns]

        return df_corr.corr()

    def fraud_distribution(self):

        count = self.df["FraudResult"].value_counts()

        return pd.DataFrame({
            "Count": count,
            "Percentage":
            round(count / len(self.df) * 100, 2)
        })

    def categorical_distribution(self, column):

        return self.df[column].value_counts()

    def numerical_columns(self):

        return self.df.select_dtypes(
            include=np.number
        ).columns.tolist()
    def outlier_report(self):

        numerical_cols = [
            "Amount",
            "Value"
        ]

        report = {}

        for col in numerical_cols:

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers =self. df[
                (self.df[col] < lower) |
                (self.df[col] > upper)
            ]

            report[col] = {
                "Outlier Count": len(outliers),
                "Outlier Percentage":
                    round(
                        len(outliers) /
                        len(self.df) * 100,
                        2
                    )
            }

        return pd.DataFrame(report).T