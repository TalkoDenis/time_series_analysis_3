from scripts.learning_model import learning_model

def test_learning_model(sample_prophet_df, mocker):
    mock_arima = mocker.patch(
        'pmdarima.auto_arima', 
        return_value="fake_model"
    )

    result = learning_model(sample_prophet_df)

    assert result == "fake_model"
    mock_arima.assert_called_once()