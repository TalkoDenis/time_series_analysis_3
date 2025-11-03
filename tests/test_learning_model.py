from stat_project.models.pmdarima import PmdarimaWrapper

def test_pmdarima_wrapper_fit(sample_prophet_df, mocker):
    fake_model = mocker.MagicMock()
    fake_model.order = (1, 0, 1)
    fake_model.seasonal_order = (0, 1, 0, 12)

    mock_auto_arima = mocker.patch(
        "pmdarima.auto_arima", 
        return_value=fake_model
    )
    
    model_wrapper = PmdarimaWrapper(m=12)
    model_wrapper.fit(sample_prophet_df)

    mock_auto_arima.assert_called_once()
    assert model_wrapper.model == fake_model