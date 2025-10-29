try:
    import stat_project as sp
except ImportError:
    print("Cannot find 'stat_project'.")
    exit(1)


def main():
    """
    Test
    """
    print("Pipline inicialization")
    pipe = sp.ForecastPipeline(path="./data/data.csv")

    try:
        print("Start")
        pipe.prepare_data(split_date="2025-01-01")
        
        print("Learning model(sarimax)")
        pipe.set_model("sarimax", m=12)
        pipe.fit()
        
        pipe.predict()

        pipe.visualize()
        
        print("First 5 strings")
        forecast = pipe.get_forecast()
        print(forecast.head())

    except Exception as e:
        print(f"Exception: {e}")


if name == "__main__":
    main()