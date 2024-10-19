import pandas as pd
from datetime import datetime, timedelta

from prediction_data_request import (
    get_forecast_wind_prod,
    get_forecast_sun_prod,
    get_forecast_elec_prices,
    get_forecast_weather,
    fmi_get_prev_x_days,
    wind_prod_get_prev_x_days,
)

from data_combiner import combine


def get_forecasts(date):
    weather_forecast = get_forecast_weather(date)
    wind_prod_forecast = get_forecast_wind_prod(date)
    print(wind_prod_forecast)
    sun_prod_forecast = get_forecast_sun_prod(date)
    print(sun_prod_forecast)
    elec_prices_forecast = get_forecast_elec_prices(date)
    return [weather_forecast, wind_prod_forecast, sun_prod_forecast, elec_prices_forecast]


def get_prev_x_days(x=5):
    weather_history = fmi_get_prev_x_days(x)
    return weather_history


def convert_columns_to_int(df):
    columns_to_convert = ['year', 'month', 'day', 'hour', 'min']
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(
                'Int64')
    if 'min' in df.columns:
        df = df.drop(columns='min')
    return df


def get_combined_forecasts(date):
    forecasts = get_forecasts(date)
    forecasts = [convert_columns_to_int(df) for df in forecasts]
    df_handler = combine(dataframes=forecasts, get_handler=True)
    df_handler.initialize(avg_temp=True, create_dates=True,
                          to_periodic=True, normalize=False, drop_columns=True, decay=False)
    return df_handler.get_initialized_dataframe()


if __name__ == "__main__":
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    print(f"tomorrow: {tomorrow}")
    combined_forecasts = get_combined_forecasts(tomorrow)

    print(combined_forecasts)
