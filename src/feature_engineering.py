import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
# from category_encoders import WOEEncoder

def create_aggregate_features(df):
    """
    Create customer-level aggregate transaction features.
    """

    agg_features = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionValue=("Value", "sum"),
            AverageTransactionValue=("Value", "mean"),
            TransactionCount=("Value", "count"),
            StdTransactionValue=("Value", "std"),
            MaxTransactionValue=("Value", "max"),
            MinTransactionValue=("Value", "min")
        )
        .reset_index()
    )

    return agg_features

def get_categorical_features():
    """
    Return categorical features to encode.
    """

    return [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]


def create_encoder():
    """
    Create One-Hot Encoder.
    """

    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


def create_encoding_pipeline():
    """
    Create categorical encoding transformer.
    """

    categorical_features = (
        get_categorical_features()
    )

    encoder = create_encoder()

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                encoder,
                categorical_features
            )
        ],
        remainder="passthrough"
    )
def handle_missing_values(df):
    """
    Handle missing values introduced during feature engineering.
    """

    if "StdTransactionValue" in df.columns:
        df["StdTransactionValue"] = (
            df["StdTransactionValue"]
            .fillna(0)
        )

    return df

def standardize_features(df, numerical_cols):
    """
    Standardize numerical features.
    """

    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    return df
# def apply_woe_encoding(
#     X_train,
#     y_train,
#     categorical_features
# ):
#     """
#     Apply WoE encoding to categorical features.
#     """

#     encoder = WOEEncoder(
#         cols=categorical_features
#     )

#     X_train_encoded = encoder.fit_transform(
#         X_train,
#         y_train
#     )

#     return X_train_encoded, encoder