import pandas as pd
import pytest
from scripts.validate_date import validate_data

def test_validate_data_success():
    good_data = pd.DataFrame({
        'date': ['2025-01-01', '2025-01-02'],
        'value': [100, 200]
    })
    
    try:
        validate_data(good_data)
    except Exception as e:
        pytest.fail(f"validate_data() make mistake on the good data: {e}")

def test_validate_data_missing_column():
    bad_data = pd.DataFrame({
        'date': ['2025-01-01', '2025-01-02'],
        'wrong_column': [100, 200]
    })
    
    with pytest.raises(Exception, match="is not valid!"):
        validate_data(bad_data)

def test_validate_data_empty_df():
    empty_data = pd.DataFrame(columns=['date', 'value'])
    
    with pytest.raises(Exception, match="is empty!"):
        validate_data(empty_data)