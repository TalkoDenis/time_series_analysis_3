import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "value": [100, 200, 300],
        }
    )


@pytest.fixture
def sample_prophet_df():
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "y": [100, 200, 300],
        }
    )
