import pmdarima as pm

def learning_model(train_df, train_param='y', exogenous_param='ds'):
    print("Start learning the model...")
    SARIMAX_model = pm.auto_arima(train[[train_param]],
                            exogenous=train[[exogenous_param]],
                            start_p=1, start_q=1,
                            test='adf',
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=None, D=1,
                            trace=False,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
    print("The Model is learnd. Params:", SARIMAX_model.order, SARIMAX_model.seasonal_order)
    return SARIMAX_model