# Stat Project: Time Series Forecasting Library

stat_project is a Python library for time series analysis and forecasting. It provides a simple, unified interface for multiple forecasting models, allowing you to easily fit, predict, and compare their results.

This project is built to be extensible, allowing new models (like Prophet or SARIMAX) to be wrapped and used with the same consistent API.

## Features

* Unified API: A single ForecastPipeline class to manage the entire workflow.
* Model Agnostic: Use pmdarima (SARIMAX) or prophet just by changing one line of code.
* Extensible: Built with an abstract BaseModel class, so you can easily add your own model wrappers.
* Data Validation: Includes robust pre-processing to validate data structure and content.
* Visualization: Built-in visualize() method to quickly plot your results with Plotly.

## Installation

This project uses uv for package management.

1.  Clone the repository:
    
    git clone [https://github.com/TalkoDenis/time_series_analysis_3.git](https://github.com/TalkoDenis/time_series_analysis_3.git)
    cd time_series_analysis_3
    

2.  Create a virtual environment:
    
    uv venv
    source .venv/bin/activate
    

3.  Install the package in editable mode (with dev dependencies):
    This command installs the stat_project library and all development tools (like pytest, ruff, `mypy`).
    
    make install
    
    *(This runs `uv pip sync --extra dev pyproject.toml`)*

## Quick Start

You can run the main example script to see the library in action. This will load the data, train a model, and show a forecast.

```bash
Example Library Usage

import stat_project as sp

pipe = sp.ForecastPipeline(path="./data/data.csv")

try:
    pipe.prepare_data(split_date="2025-01-01") \
        .set_model("sarimax", m=12) \
        .fit() \
        .predict()

    pipe.visualize()

    forecast_df = pipe.get_forecast()
    print(forecast_df.head())

    print("--- Now trying Prophet ---")
    pipe.set_model("prophet") \
        .fit() \
        .predict() \
        .visualize()