from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.feature_engineering import (
    create_aggregate_features,
    handle_missing_values,
    get_categorical_features,
    create_encoding_pipeline,
    standardize_features
)

class DropColumnsTransformer(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        columns_to_drop = [
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "CustomerId",
            "CurrencyCode",
            "TransactionStartTime"
        ]

        return X.drop(
            columns=columns_to_drop,
            errors="ignore"
        )
class FeatureEngineeringPipeline(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # 1. Aggregate Features
        customer_features = create_aggregate_features(X)

        X = X.merge(
            customer_features,
            on="CustomerId",
            how="left"
        )

        # 2. Missing Values
        X = handle_missing_values(X)

        # 3. Drop non-predictive columns BEFORE encoding
        columns_to_drop = [
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "CustomerId",
            "CurrencyCode",
            "TransactionStartTime"
        ]

        X = X.drop(
            columns=columns_to_drop,
            errors="ignore"
        )

        # 4. Encoding
        encoder = create_encoding_pipeline()

        encoded = encoder.fit_transform(X)

        feature_names = (
            encoder.get_feature_names_out()
        )

        X = pd.DataFrame(
            encoded,
            columns=feature_names,
            index=X.index
        )

        return X
class ScalingTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):

        self.scaler.fit(X)

        return self

    def transform(self, X):

        X_scaled = self.scaler.transform(X)

        return pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index
        )


def create_data_processing_pipeline():

    pipeline = Pipeline([
        (
            "feature_engineering",
            FeatureEngineeringPipeline()
        ),
        (
            "standardization",
            ScalingTransformer()
        )
    ])

    return pipeline