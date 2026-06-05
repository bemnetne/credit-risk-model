from pydantic import BaseModel


class PredictionRequest(BaseModel):

    TotalTransactionValue: float
    AverageTransactionValue: float
    TransactionCount: float
    StdTransactionValue: float
    MaxTransactionValue: float
    MinTransactionValue: float


class PredictionResponse(BaseModel):

    risk_probability: float

    credit_score: int

    recommended_limit: float

    recommended_duration_days: int