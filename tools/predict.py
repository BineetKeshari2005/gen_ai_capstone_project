"""
Tool 1: predict_churn
Wraps the XGBoost pipeline to predict customer churn.
Returns prediction label and churn probability — no business logic.
"""

import json
import warnings
import joblib
import pandas as pd
from langchain_core.tools import tool

from src.config import PIPELINE_PATH

warnings.filterwarnings("ignore")

LABEL_MAP = {1: "Will Churn", 0: "Will Stay"}


@tool
def predict_churn(customer_data: str) -> str:
    """Predict whether a customer will churn or not.

    Args:
        customer_data: A JSON string containing customer attributes:
            Tenure, CityTier, WarehouseToHome, HourSpendOnApp,
            NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress,
            Complain, OrderAmountHikeFromlastYear, CouponUsed,
            OrderCount, DaySinceLastOrder, CashbackAmount,
            PreferredLoginDevice, PreferredPaymentMode, Gender,
            PreferedOrderCat, MaritalStatus.

    Returns:
        JSON string with prediction (0 or 1), churn_probability,
        confidence_percent, and human-readable label.
    """
    data = json.loads(customer_data)
    pipeline = joblib.load(PIPELINE_PATH)
    df = pd.DataFrame([data])

    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0][1])

    result = {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "confidence_percent": round(probability * 100, 2),
        "label": LABEL_MAP[prediction],
    }

    return json.dumps(result)
