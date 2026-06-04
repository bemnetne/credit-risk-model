import pandas as pd

from src.feature_engineering import (
    create_aggregate_features,handle_missing_values
)


def test_create_aggregate_features_columns():

    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "Value": [100, 200, 300]
    })

    result = create_aggregate_features(df)

    expected_columns = [
        "CustomerId",
        "TotalTransactionValue",
        "AverageTransactionValue",
        "TransactionCount",
        "StdTransactionValue",
        "MaxTransactionValue",
        "MinTransactionValue"
    ]

    assert list(result.columns) == expected_columns

def test_handle_missing_values():

    df = pd.DataFrame({
        "StdTransactionValue": [10.5, None, 5.2]
    })

    result = handle_missing_values(df)

    assert result["StdTransactionValue"].isna().sum() == 0
    assert result.loc[1, "StdTransactionValue"] == 0