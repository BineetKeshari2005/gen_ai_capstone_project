"""
Tool 3: get_customer_segment_stats
Reads the original dataset and computes average feature values
for churned vs retained customers, giving the LLM benchmark context.
"""

import json
import warnings
import pandas as pd
from langchain_core.tools import tool

from src.config import DATASET_PATH, NUMERICAL_FEATURES

warnings.filterwarnings("ignore")


@tool
def get_customer_segment_stats(query: str) -> str:
    """Get average statistics comparing churned vs retained customer segments.

    Reads the original e-commerce dataset and computes mean values
    of all numerical features grouped by churn status (0=Retained, 1=Churned).
    This provides benchmark comparisons to contextualize any individual customer.

    Args:
        query: Any string (this tool doesn't need specific input,
               it always returns the full segment comparison).

    Returns:
        JSON string with average feature values for churned and
        retained segments, plus the churn rate.
    """
    df = pd.read_excel(DATASET_PATH, sheet_name="E Comm")

    # Compute segment averages grouped by Churn status
    segment_means = (
        df.groupby("Churn")[NUMERICAL_FEATURES]
        .mean()
        .round(2)
        .to_dict(orient="index")
    )

    total_customers = len(df)
    churned_count = int(df["Churn"].sum())
    retained_count = total_customers - churned_count

    result = {
        "total_customers": total_customers,
        "churned_count": churned_count,
        "retained_count": retained_count,
        "churn_rate_percent": round(churned_count / total_customers * 100, 2),
        "segment_averages": {
            "retained (Churn=0)": segment_means.get(0, {}),
            "churned (Churn=1)": segment_means.get(1, {}),
        },
    }

    return json.dumps(result, indent=2)